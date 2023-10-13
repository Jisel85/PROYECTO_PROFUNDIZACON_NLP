from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())


with open("data/text_to_test.txt", "r") as doc:
    text_to_summarize = doc.read()

template = """Dame el resumen en idioma español.
Resume el siguiente texto:
{text}
Enumera los 5 principales hechos.
HECHOS ENUMERADOS:"""

repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
llm = HuggingFaceHub(
    repo_id=repo_id,
    verbose=True,
    model_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 1024,
        "max_length": 1024,
        "task": "summarization"
    }
)
prompt = PromptTemplate.from_template(template=template)
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run(text_to_summarize))
