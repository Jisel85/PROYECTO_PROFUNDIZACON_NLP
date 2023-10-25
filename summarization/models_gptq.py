import gc
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from utils.summary import count_tokens_mistral_base_llm, get_template_to_summary_llms_mistral_base, get_list_text_recursive_splitter


models_gptq = [
    "TheBloke/dolphin-2.1-mistral-7B-GPTQ""",
    "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",
    "TheBloke/Mistral-7B-OpenOrca-GPTQ",
    "TheBloke/samantha-1.2-mistral-7B-GPTQ",
    "TheBloke/SlimOpenOrca-Mistral-7B-GPTQ",
    "TheBloke/zephyr-7B-alpha-GPTQ",
]


def create_hf_pipe_summarization(model_name="TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"):
    """
    Cree una canalización de Hugging Face en modelos LLM para usar en la tarea de resumen.
    :param model_name: Model name in the style of HuggingFace hub. (Ex: TheBloke/Mistral-7B-Instruct-v0.1-GPTQ)
    :return: pipeline
    """
    revision = "gptq-4bit-32g-actorder_True"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=False,
        revision=revision
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    generation_config = GenerationConfig.from_pretrained(model_name)
    generation_config.max_new_tokens = 512
    generation_config.temperature = 0.7
    generation_config.do_sample = True
    generation_config.top_k = 40
    generation_config.top_p = 0.95
    generation_config.repetition_penalty = 1.1

    hf_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
    )

    return hf_pipe


def get_simple_summary_gptq(text, model_name="TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"):
    """
    :param text:
    :param model_name:
    :return:
    """
    hf_pipe = create_hf_pipe_summarization(model_name=model_name)
    prompt_template = get_template_to_summary_llms_mistral_base(text, model_name="mistral")
    summary_llm = hf_pipe(prompt_template)[0]['generated_text'].split("[/INST]")[-1]

    return summary_llm


def get_map_summary_gptq(text, model_name="TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"):
    """
    This function returns a summary of the document in sections, that is, the document is divided into pieces and
    generated summary of each piece. Returns a list with the summary of each chunk. The summaries follow the order of
    the occurrence in the document. Models options: models_gptq
    :param text: Recibe el texto raw como string.
    :param model_name: 
    :return:
    """
    hf_pipe = create_hf_pipe_summarization(model_name=model_name)

    list_text_splitter, map_summary = get_list_text_recursive_splitter(text), []

    for text_splitter in list_text_splitter:
        prompt_template = get_template_to_summary_llms_mistral_base(text_splitter, model_name="mistral")
        print(text_splitter)
        summary_llm = hf_pipe(prompt_template)[0]['generated_text'].split("[/INST]")[-1]
        print("Summary:", summary_llm)
        map_summary.append(summary_llm)

    del hf_pipe
    gc.collect()
    time.sleep(5)
    return map_summary


def get_summary_from_gptq(text: str, model_name="TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"):
    """
    :param text:
    :param model_name:
    :return:
    """
    if count_tokens_mistral_base_llm(text) < 1200:
        simple_summary = get_simple_summary_gptq(text, model_name=model_name)
        return simple_summary

    map_summary = get_map_summary_gptq(text, model_name=model_name)
    doc = "\n\n".join(map_summary)
    print(doc)
    print("Tokens:", count_tokens_mistral_base_llm(doc))

    hf_pipe = create_hf_pipe_summarization(model_name=model_name)
    prompt_template = get_template_to_summary_llms_mistral_base(text, model_name="mistral")
    summary_llm = hf_pipe(prompt_template)[0]['generated_text'].split("[/INST]")[-1]

    return summary_llm
