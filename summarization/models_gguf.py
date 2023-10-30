from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from utils.summary import (
    count_tokens_mistral_base_llm,
    get_list_text_recursive_splitter,
    get_template_to_summarization_llm,
)


def create_gguf_llm(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
):
    """
    :param repo_id: Model name in huggigface hub
    :param filename: Specific .gguf file to use as llm.
    :return:
    """
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=40,
        n_ctx=2048,
        max_tokens=1024,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=True,
    )

    return llm


def get_simple_summary(
    text,
    repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
):
    """
    :param text: Text to summarize.
    :param repo_id: Model name in huggingface hub (Ex: TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
    :param filename: Name of the particular gguf file to use. (Ex: mistral-7b-instruct-v0.1.Q4_K_M.gguf)
    :return:
    """
    system_message = "Eres un asistente que realiza resumenes de alta calidad."
    user_message = """Realiza un resumen conciso del siguiente texto: {text}.
    Tu respuesta final debe ser en espanol""".format(
        text=text
    )
    llm = create_gguf_llm(repo_id=repo_id, filename=filename)
    template = get_template_to_summarization_llm(repo_id=repo_id)
    prompt = PromptTemplate(
        template=template, input_variables=["system_message", "user_message"]
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    llm_answer = llm_chain.run(system_message=system_message, user_message=user_message)

    return llm_answer


def get_map_summary(
    text,
    repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
):
    """
    This function returns a summary of the document in sections, that is, the document is divided into chunks and
    generated summary of each chunk. Returns a list with the summary of each chunk. The summaries follow the order of
    the occurrence in the document.
    :param text: Long text to summarize.
    :param repo_id: Model name in huggingface hub (Ex: TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
    :param filename: Name of the particular gguf file to use. (Ex: mistral-7b-instruct-v0.1.Q4_K_M.gguf)
    :return:
    """
    list_text_splitter, map_summary = get_list_text_recursive_splitter(text), []

    for text_splitter in list_text_splitter:
        response = get_simple_summary(
            text=text_splitter, repo_id=repo_id, filename=filename
        )
        map_summary.append(response)

    return map_summary


def get_summary_from_gguf_llm(
    text: str,
    repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
):
    """
    :return:
    """
    if count_tokens_mistral_base_llm(text) < 1200:
        simple_summary = get_simple_summary(text)
        return simple_summary

    map_summary = get_map_summary(text, repo_id=repo_id, filename=filename)
    doc = "\n\n".join(map_summary)
    print(doc)
    print("Tokens:", count_tokens_mistral_base_llm(doc))
    #  It remains to be resolved if the mapping exceeds 1200 tokens. We can apply again: get_map_summary!!!!
    text_summary = get_simple_summary(doc, repo_id=repo_id, filename=filename)

    return text_summary
