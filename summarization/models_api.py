import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain


set_llm_cache(InMemoryCache())
load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_docs_recursive_splitter(string_text: str, model_name="mistral"):
    """
    Gets the text of the context size according to the structure that will be used for the summary.
    :return:
    """
    mapper_chunck_size = {
        "openai": 10000,
        "huggingface_api": 5000,
    }

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n"],
        chunk_size=mapper_chunck_size[model_name],
        chunk_overlap=mapper_chunck_size[model_name] * 0.05,  # 5% overlapping.
        add_start_index=True,
    )
    list_text_splitter = text_splitter.create_documents([string_text])

    return list_text_splitter


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
        verbose=True  # View in console the process
    )
    return summary_chain


def get_summary_from_openai(text):
    """
    :param text:
    :return:
    """
    llm = ChatOpenAI(temperature=0)
    docs = get_docs_recursive_splitter(text, model_name="openai")
    summary_chain = create_summarization_chain(llm)

    return summary_chain.run(docs)


def get_summary_from_huggingface(text, model_name="HuggingFaceH4/zephyr-7b-alpha"):
    """
    We obtain summaries of long documents using the Huggingface inference API (Requires API KEY)
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
    docs = get_docs_recursive_splitter(text, model_name="huggingface_api")
    summary_chain = create_summarization_chain(llm)

    return summary_chain.run(docs)
