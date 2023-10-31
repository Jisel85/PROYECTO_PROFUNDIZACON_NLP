from utils.gptq_processor import GPTQProcessor
from utils.summary import get_list_text_recursive_splitter


def get_summary_from_gptq(text: str, model_name="TheBloke/Mistral-7B-Instruct-v0.1-GPTQ", revision="main"):
    """
    :param revision:
    :param text:
    :param model_name:
    :return:
    """
    gptq = GPTQProcessor(model_name=model_name, revision=revision)
    map_summary = gptq.batch_process(get_list_text_recursive_splitter(text))
    summary_llm = gptq.process_summary(map_summary)

    return summary_llm
