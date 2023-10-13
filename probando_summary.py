from utils.judgments import MongoJudgments
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain import HuggingFacePipeline
from transformers import pipeline, AutoModelForSeq2SeqLM, MT5Tokenizer

model_path = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = MT5Tokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path
)


pipeline = pipeline(
    task="summarization",
    model=model_path,
    tokenizer=tokenizer,
    # max_length=1024
    # max_new_tokens=1024
)

llm = HuggingFacePipeline(
    model_id=model_path,
    pipeline=pipeline,
    verbose=True,
)
##
text = MongoJudgments("T-672-97").get_feature_of_judgment("raw_text")
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"],
    chunk_size=2500,
    chunk_overlap=250
)

docs = text_splitter.create_documents([text])

map_prompt = """La respuesta debe estar en español.
Escribe un resumen conciso de lo siguiente:
{text}
RESUMEN CONCISO:"""

combine_prompt = """La respuesta debe estar en español.
Escribe un resumen conciso de lo siguiente:
Retorna tu respuesta con numerales los cuales cubren los puntos claves del texto.
{text}
RESUMEN CON NUMERALES:"""

map_prompt_template = PromptTemplate.from_template(map_prompt)
combine_prompt_template = PromptTemplate.from_template(combine_prompt)

summary_chain = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",
    map_prompt=map_prompt_template,
    combine_prompt=combine_prompt_template,
    verbose=True
)

print(summary_chain.run(docs))


import logging
from utils.judgments import MongoJudgments
from langchain import HuggingFacePipeline
from transformers import BartTokenizer, BartForCausalLM, pipeline, BartForConditionalGeneration
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain

logging.info(f"Star")
raw_text_judgment = MongoJudgments("T-006-10").get_feature_of_judgment("raw_text")
text = MongoJudgments("T-672-97").get_feature_of_judgment("raw_text")

# llm = ChatOpenAI(temperature=0)

model_path = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = BartTokenizer.from_pretrained(model_path, model_max_length=512)
model = BartForConditionalGeneration.from_pretrained(
    model_path
)

pipeline = pipeline(
    task="summarization",
    model=model_path,
    tokenizer=tokenizer,
    max_length=512
)

llm = HuggingFacePipeline(
    pipeline=pipeline,
    verbose=True,
    # model_kwargs={
    #     "max_length": 512
    # }
)

# Map
map_template = """The following is a set of documents
{docs}
Based on this list of docs, please identify the main themes 
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt, )
# Reduce
reduce_template = """La respuesta debe estar en español.
The following is set of summaries:
{doc_summaries}
Take these and distill it into a final, consolidated summary of the main themes. 
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)
# Run chain
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain,
    document_variable_name="doc_summaries",
    verbose=True
)

# Combines and iteravely reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=512,
    verbose=True
)
# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)

text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=450,
    chunk_overlap=50
)

docs = text_splitter.create_documents([text])
print(map_reduce_chain.run(docs))
logging.info(f"End")
