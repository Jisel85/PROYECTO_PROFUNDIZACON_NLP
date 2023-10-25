import os
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def count_tokens_mistral_base_llm(text: str) -> int:
    """Returns the number of tokens in a text string with mistral embeddings."""
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1",
    )
    return len(tokenizer.encode(text))


def count_tokens_from_openai(string: str) -> int:
    """Returns the number of tokens in a text string with openai embeddings."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))


def get_template_to_summary_llms_mistral_base(text, model_name="mistral"):
    """
    Obtain the prompt-input to the models based on Mistral-7b
    :param text:
    :param model_name:
    :return:
    """
    dict_prompts_template = {
        "mistral": """<s>[INST]Eres un asistente que realiza resumenes de alta calidad.[/INST]</s>[INST]
        Realiza un resumen conciso y en espanol del siguiente texto: {} [/INST]""",
        "zephyr": """<|system|> 
        Eres un asistente que realiza resumenes de alta calidad. </s>
        <|user|> Realiza un resumen conciso del siguiente texto: {}.</s><|assistant|>""",
        "openorca": """<|im_start|> system Eres un asistente que realiza resumenes de alta calidad.<|im_start|>user 
        Realiza un resumen conciso del siguiente texto: {} Proporciona tu respuesta en 
        espanol.<|im_end|><|im_start|>assistant"""
    }

    return dict_prompts_template[model_name].format(text)


def get_list_text_recursive_splitter(string_text: str, model_name="mistral"):
    """
    Gets the text of the context size according to the structure that will be used for the summary.
    :return:
    """
    mapper_chunck_size = {
        "openai": 10000,
        "gpt4all": 6500,
        "mistral": 5000,
        "huggingface_api": 5000,
    }

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n"],
        chunk_size=mapper_chunck_size[model_name],
        chunk_overlap=mapper_chunck_size[model_name] * 0.05,  # 5% overlapping.
        add_start_index=True,
    )
    list_text_splitter = text_splitter.split_text(string_text)

    return list_text_splitter
