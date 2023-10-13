import torch
import logging
import tiktoken
from utils.judgments import MongoJudgments
from transformers import RobertaTokenizerFast, EncoderDecoderModel
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_summary_roberta(text):
    """
    :param text:
    :return:
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = "Narrativa/bsc_roberta2roberta_shared-spanish-finetuned-mlsum-summarization"
    tokenizer = RobertaTokenizerFast.from_pretrained(model)
    model = EncoderDecoderModel.from_pretrained(model).to(device)
    inputs = tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = model.generate(input_ids, attention_mask=attention_mask)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def num_tokens_from_string(string: str, encoding_name="gpt-3.5-turbo") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_docs_judgment_to_openai(judgment):
    """
    :param str judgment: id_judgment
    :return:
    """
    text_raw_judgment = MongoJudgments(judgment).get_feature_of_judgment(feature="raw_text")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=10000,
        chunk_overlap=500,
        add_start_index=True,
    )
    docs = text_splitter.create_documents([text_raw_judgment])
    return docs


def get_summarization_from_openai(judgment):
    """
    :param judgment:
    :return:
    """
    llm = ChatOpenAI(temperature=0)
    docs = get_docs_judgment_to_openai(judgment)

    map_prompt = """La respuesta debe estar en español.
    Escribe un resumen conciso de lo siguiente:
    {text}
    RESUMEN CONCISO:"""

    combine_prompt = """La respuesta debe estar en español.
    Escribe un resumen conciso de lo siguiente:
    {text}
    Retorna tu respuesta con numerales los cuales cubren los puntos claves del texto.    
    RESUMEN CON NUMERALES:"""

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=PromptTemplate.from_template(map_prompt),
        combine_prompt=PromptTemplate.from_template(combine_prompt),
        # verbose=True
    )
    return summary_chain.run(docs)
