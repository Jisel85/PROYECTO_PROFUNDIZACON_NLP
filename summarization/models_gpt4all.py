import gc
import time
from pathlib import Path
from gpt4all import GPT4All
from huggingface_hub import hf_hub_download
from utils.summary import get_template_to_summary_llms_mistral_base, count_tokens_mistral_base_llm, \
    get_list_text_recursive_splitter


def create_model_llm(model_name="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                     filename="mistral-7b-instruct-v0.1.Q4_0.gguf"):
    """
    :param filename:
    :param model_name:
    :return:
    """
    model_path = hf_hub_download(repo_id=model_name, filename=filename)
    llm = GPT4All(
        model_name=Path(model_path).name,
        model_path=Path(model_path).parent,
        device="gpu",
    )

    return llm


def get_simple_summary_gpt4all(text, llm):
    """
    :param llm:
    :param text:
    :return:
    """
    simple_summary = llm.generate(
        prompt=get_template_to_summary_llms_mistral_base(text),
        max_tokens=512,
        temp=0.01,
    )
    return simple_summary


def get_map_summary_gpt4all(text, llm):
    """
    This function returns a summary of the document in sections, that is, the document is divided into pieces and
    generated summary of each piece. Returns a list with the summary of each chunk. The summaries follow the order of
    the occurrence in the document. Models options: mistral-7b-openorca, zephyr-7b-alpha, mistral-7b-instruct-v0.1
    :param text:
    :param model_name:
    :return:
    """
    list_text_splitter = get_list_text_recursive_splitter(text)

    map_summary_string = ""
    for text_splitter in list_text_splitter:
        response = llm.generate(
            prompt=get_template_to_summary_llms_mistral_base(text_splitter),
            max_tokens=512,
            temp=0.01,
        )
        map_summary_string += "\n\n" + response
    del llm, response
    gc.collect()
    time.sleep(5)

    return map_summary_string


def get_summary_from_gpt4all(text: str, model_name="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                             filename="mistral-7b-instruct-v0.1.Q4_0.gguf"):
    """
    :param text:
    :param filename:
    :param model_name:
    :return:
    """
    llm = create_model_llm(model_name=model_name, filename=filename)

    if count_tokens_mistral_base_llm(text) < 1200:
        return get_simple_summary_gpt4all(text)

    map_summary_string = get_map_summary_gpt4all(text, llm)
    #  It remains to be resolved if the mapping exceeds 1200 tokens!!!!
    text_summary = llm.generate(
        prompt=get_template_to_summary_llms_mistral_base(text=map_summary_string, model_name="mistral"),
        max_tokens=1024,
        temp=0.01,
    )
    del llm
    gc.collect()
    return text_summary
