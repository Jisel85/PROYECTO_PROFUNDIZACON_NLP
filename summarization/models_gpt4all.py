import os
import gc
import time
import torch
from pathlib import Path
from gpt4all import GPT4All
from utils.summary import get_template_to_summary_llms_mistral_base, count_tokens_mistral_base_llm, \
    get_list_text_recursive_splitter

PROJECT_PATH = Path(os.getcwd())
PATH_MODELS = PROJECT_PATH / "models" / "huggingface" / "hub"

torch.cuda.empty_cache()


def create_model_llm(model_name="mistral-7b-instruct-v0.1"):
    """
    :param model_name:
    :return:
    """
    llm = GPT4All(
        model_name=f"{model_name}.Q4_0.gguf",
        model_path=PATH_MODELS,
        device="gpu",
        allow_download=False
    )

    return llm


def get_simple_summary_gpt4all(text, model_name="mistral-7b-instruct-v0.1"):
    """
    :param model_name:
    :param text:
    :return:
    """
    llm = create_model_llm(model_name=model_name)
    simple_summary = llm.generate(
        prompt=get_template_to_summary_llms_mistral_base(text),
        max_tokens=512,
        temp=0.01,
    )
    return simple_summary


def get_map_summary_gpt4all(text, model_name="mistral-7b-instruct-v0.1"):
    """
    This function returns a summary of the document in sections, that is, the document is divided into pieces and
    generated summary of each piece. Returns a list with the summary of each chunk. The summaries follow the order of
    the occurrence in the document. Models options: mistral-7b-openorca, zephyr-7b-alpha, mistral-7b-instruct-v0.1
    :param text:
    :param model_name:
    :return:
    """
    llm = create_model_llm(model_name=model_name)
    list_text_splitter, map_summary = get_list_text_recursive_splitter(text), []

    for text_splitter in list_text_splitter:
        response = llm.generate(
            prompt=get_template_to_summary_llms_mistral_base(text_splitter),
            max_tokens=512,
            temp=0.01,
        )
        map_summary.append(response)
    del llm, response
    gc.collect()
    time.sleep(5)

    return map_summary


def get_summary_from_gpt4all(text: str, model_name="mistral-7b-instruct-v0.1"):
    """
    :param text:
    :param model_name:
    :return:
    """
    if count_tokens_mistral_base_llm(text) < 1200:
        simple_summary = get_simple_summary_gpt4all(text, model_name=model_name)
        return simple_summary

    map_summary = get_map_summary_gpt4all(text, model_name=model_name)
    doc = "\n\n".join(map_summary)
    #  It remains to be resolved if the mapping exceeds 1200 tokens!!!!
    llm = create_model_llm(model_name=model_name)
    text_summary = llm.generate(
        prompt=get_template_to_summary_llms_mistral_base(text=doc, model_name="mistral"),
        max_tokens=1024,
        temp=0.01,
    )

    return text_summary
