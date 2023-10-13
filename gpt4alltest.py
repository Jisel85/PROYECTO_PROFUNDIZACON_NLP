import os
import tiktoken
from datetime import datetime
from pathlib import Path
from transformers import LlamaTokenizerFast
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

set_llm_cache(InMemoryCache())

# API_KEY_HUGGINGFACE = os.environ["HUGGINGFACEHUB_API_TOKEN"]
MAIN_PATH_MODELS_GPT4ALL = Path("/home/andres-campos/.cache/gpt4all/")
tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-7b-chat-hf")


def num_tiktoken_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_summary_from_llm_gpt4all(text, model_name="orca-mini-3b", device="NVIDIA GeForce RTX 3060 Laptop GPU"):
    """
    Model support: llama-2-7b-chat, orca-mini-13b, orca-mini-7b, orca-mini-3b, gpt4all-falcon, GPT4All-13B-snoozy,
    wizardlm-13b-v1.1-superhot-8k
    :param device:
    :param str model_name:
    :param str text:
    :return:
    """
    print(f"Count tokens with LlamaTokenizerFast in text : {len(tokenizer.encode(text))}")
    print(f"Count tokens with tiktoken library in text: {num_tiktoken_from_string(text)}")
    if model_name == "gpt4all-falcon":
        path_model = MAIN_PATH_MODELS_GPT4ALL / "ggml-model-gpt4all-falcon-q4_0.bin"
    else:
        path_model = MAIN_PATH_MODELS_GPT4ALL / f"{model_name}.ggmlv3.q4_0.bin"

    template = """Give me a summary of the following text:
    {text}
    List the top 5 facts.
    """
    prompt = PromptTemplate(template=template, input_variables=["text"])
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GPT4All(
        model=str(path_model),
        callbacks=callbacks,
        verbose=True,  # Verbose is required to pass to the callback manager
        n_threads=os.cpu_count(),
        device=device
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain.run(text)


with open("data/text_to_test.txt") as file:
    text_to_test = file.read()

print(datetime.now())
get_summary_from_llm_gpt4all(text_to_test, model_name="orca-mini-3b")
print(datetime.now())