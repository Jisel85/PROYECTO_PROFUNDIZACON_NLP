from utils.judgments import MongoJudgments
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain


def get_docs_from_judgment(judgment):
    """
    :param str judgment: id_judgment
    :return:
    """
    text_raw_judgment = MongoJudgments(judgment).get_feature_of_judgment(feature="raw_text")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True,
    )
    docs = text_splitter.create_documents([text_raw_judgment])
    return docs


def get_summary_judgment(judgment):
    """
    :return: str Summary from judgment
    """
    docs = get_docs_from_judgment(judgment)
    prompt_template = """Tu respuesta debe estar es español.
    Escribe un resumen conciso de lo siguiente:
    "{text}"
    Dame también los 5 temas central enumerados.
    RESUMEN CONCISO:"""

    prompt = PromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="text"
    )
    judgment_summary = stuff_chain.run(docs)

    return judgment_summary


client_judgment = MongoJudgments("T-672-97")
summary = get_summary_judgment(judgment=client_judgment.judgment)
print(summary)
client_judgment.write_judgment_feature(content=summary, feature="summary_gpt-3_5-turbo-16k")
