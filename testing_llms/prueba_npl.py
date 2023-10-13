import langchain

#langchain.__version__

import torch
from langchain import HuggingFacePipeline

torch.cuda.empty_cache()


llm = HuggingFacePipeline.from_model_id(
    model_id="OpenAssistant/stablelm-7b-sft-v7-epoch-3", 
    task="text-generation",
    device=0,
    model_kwargs={"temperature": 0.0, "max_length": 1024, "torch_dtype": torch.float16,}
)

from langchain import PromptTemplate
from langchain import LLMChain

template = """<|prompter|>{question}<|endoftext|><|assistant|>"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.run('entiendes español? dios existe?'))