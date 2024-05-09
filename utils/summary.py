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
        token=HUGGINGFACEHUB_API_TOKEN
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
feature/testing_llms
        "mistral": """<s>[INST]Eres un asistente que realiza resumenes en espanol de alta calidad.[/INST]</s>[INST]
        Realiza un resumen conciso y en espanol del siguiente texto: {} [/INST]""",
        #"zephyr": """<|system|> 
        #Eres un asistente que realiza resumenes en espanol de alta calidad. </s>
        #<|user|> Realiza un resumen en espanol conciso del siguiente texto: {}.</s><|assistant|>""",
        "zephyr": """<|system|> 
        Please provide a summary of the text in Spanish: </s>
        <|user|> Please provide a summary of the text in Spanish:: {}.</s><|assistant|>""", 
        "openorca": """<|im_start|> system Eres un asistente que realiza resumenes en espanol de alta calidad.<|im_start|>user 
=======
        "mistral": """<s>[INST]Eres un asistente que realiza resumenes de alta calidad.[/INST]</s>[INST]
        Realiza un resumen conciso y en espanol del siguiente texto: {} [/INST]""",
        "zephyr": """<|system|> 
        Eres un asistente que realiza resumenes de alta calidad. </s>
        <|user|> Realiza un resumen conciso del siguiente texto: {}.</s><|assistant|>""",
        "openorca": """<|im_start|> system Eres un asistente que realiza resumenes de alta calidad.<|im_start|>user 
main
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


# ===================================== To use in GGUF models ======================================================= #
mistral = """<s>[INST]
{system_message}[/INST]</s>
[INST]
{user_message}[/INST]"""

zephyr = """<|system|>
{system_message}</s>
<|user|>
{user_message}</s>
<|assistant|>"""

openorca = """<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""

dict_models_template = {
    "TheBloke/Mistral-7B-Instruct-v0.1-GGUF": mistral,
    "TheBloke/dolphin-2.1-mistral-7B-GGUF": openorca,
    "TheBloke/SlimOpenOrca-Mistral-7B-GGUF": openorca,
    "TheBloke/zephyr-7B-alpha-GGUF": zephyr,
    "TheBloke/zephyr-7B-beta-GGUF": zephyr,
    "TheBloke/samantha-1.2-mistral-7B-GGUF": openorca,
}


def get_template_to_summarization_llm(repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF"):
    """
    :param repo_id:  Name of the model in Huggingface hub (Ex: TheBloke/zephyr-7B-beta-GGUF )
    :return:
    """
    return dict_models_template[repo_id]
