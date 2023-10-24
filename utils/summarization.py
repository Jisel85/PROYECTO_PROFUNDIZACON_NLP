import os
import torch
import logging
import tiktoken
from pathlib import Path
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All, HuggingFaceHub, HuggingFacePipeline
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizerFast,
    GenerationConfig,
)

set_llm_cache(InMemoryCache())
torch.cuda.empty_cache()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
MAIN_PATH_MODELS_GPT4ALL = Path("/home/andres-campos/.cache/gpt4all/")

mapper_chunck_size = {
    "openai": 10000,
    "gpt4all": 6500,
    "mistral_base": 2000,
    "huggingface_api": 5000,
}


def count_tokens_from_openai(string: str) -> int:
    """Returns the number of tokens in a text string with openai embeddings."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))


def count_tokens_from_gpt4all(text: str) -> int:
    """Returns the number of tokens in a text string with LlamA embeddings."""
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
        token=HUGGINGFACEHUB_API_TOKEN
    )
    return len(tokenizer.encode(text))


def get_documents_from_judgment(string_text, model="openai"):
    """
    Gets the judgment documents of the context size according to the structure that will be used for the summary
    :return:
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n"],
        chunk_size=mapper_chunck_size[model],
        chunk_overlap=mapper_chunck_size[model] * 0.05,  # Dejemos el 5% de overlapping.
        add_start_index=True,
    )
    docs = text_splitter.create_documents([string_text])
    return docs


def create_summarization_chain(llm):
    """Create a summary chain to apply map reduce"""
    # Generals templates of summarize judgment #
    map_prompt = """Your final answer must be in Spanish.
    Write a concise summary of the following:
    {text}
    RESUMEN CONCISO:"""

    combine_prompt = """The answer must be in Spanish.
    Write a concise summary of the following:
    {text}
    Return your answer which cover the key points of the text.   
    SUMMARY IN SPANISH WITH NUMERALS:"""
    # Generals templates of summarize judgment #

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=PromptTemplate.from_template(map_prompt),
        combine_prompt=PromptTemplate.from_template(combine_prompt),
        verbose=True
    )
    return summary_chain


def get_summary_from_openai(text):
    """
    :param text:
    :return:
    """
    llm = ChatOpenAI(temperature=0)
    docs = get_documents_from_judgment(text, model="openai")
    summary_chain = create_summarization_chain(llm)

    return summary_chain.run(docs)


def get_summary_from_llm_gpt4all(text, model_name="llama-2-7b-chat", device="cpu"):
    """
    This function obtain summary of judgment using GPT4ALL models (https://gpt4all.io/index.html).
    Model support: llama-2-7b-chat, orca-mini-13b, orca-mini-7b, orca-mini-3b, gpt4all-falcon, GPT4All-13B-snoozy,
    wizardlm-13b-v1.1-superhot-8k (Recommended)

    :param str text: Text like a string to summarize.
    :param model_name:
    :param device: Optional. Name your GPU, for example: NVIDIA GeForce RTX 3060 Laptop GPU
    :return:
    """
    if model_name == "gpt4all-falcon":
        path_model = MAIN_PATH_MODELS_GPT4ALL / "ggml-model-gpt4all-falcon-q4_0.bin"
    else:
        path_model = MAIN_PATH_MODELS_GPT4ALL / f"{model_name}.ggmlv3.q4_0.bin"

    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GPT4All(
        model=str(path_model),
        callbacks=callbacks,
        max_tokens=1024,
        verbose=True,  # Verbose is required to pass to the callback manager
        n_threads=os.cpu_count(),
        device=device
    )
    docs = get_documents_from_judgment(text, model="gpt4all")
    summary_chain = create_summarization_chain(llm)

    return summary_chain.run(docs)


def get_summary_from_huggingface_api(text, model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    """
    Obtenemos resumenes de documentos largos haciendo uso de la API inference de Huggingface (Necesita API KEY)
    :param str text: Large text to be summarize.
    :param str model_name: Model name to use from huggingface host.
    :return: Summary of the tetx.
    """
    llm = HuggingFaceHub(
        task="text-generation",
        repo_id=model_name,
        verbose=True,
        model_kwargs={
            "temperature": 0.001,
            "max_new_tokens": 600,
        },
    )
    docs = get_documents_from_judgment(text, model="huggingface_api")
    summary_chain = create_summarization_chain(llm)

    return summary_chain.run(docs)


def get_summary_from_llm_mistral_base(text, model_name="HuggingFaceH4/zephyr-7b-alpha"):
    """
    :param text:
    :param model_name:
    :return:
    """
    generation_config = GenerationConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    generation_config.max_new_tokens = 256
    generation_config.temperature = 0.001
    generation_config.do_sample = True
    generation_config.top_k = 50
    generation_config.top_p = 0.95

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        return_full_text=True,
    )

    llm = HuggingFacePipeline(pipeline=pipe, verbose=True)
    docs = get_documents_from_judgment(text, model="mistral_base")
    summary_chain = create_summarization_chain(llm)

    return summary_chain.run(docs)
