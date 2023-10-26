#-------->> solucion de chatgpt4  
import gc
import time
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

def count_tokens_from_gguf_models(text: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1"
    )
    return len(tokenizer.encode(text))

def get_template_to_summary_llms_mistral_base(text, model_name="mistral"):
    dict_prompts_template = {
        "mistral": """<s>[INST]Eres un asistente que realiza resumenes de alta calidad.[/INST]</s>[INST]
        Realiza un resumen conciso y en espanol del siguiente texto: {} [/INST]""",
        "zephyr": """
        Eres un asistente que realiza resumenes de alta calidad. </s>
         Realiza un resumen conciso del siguiente texto: {}.</s>""",
        "openorca": """ system Eres un asistente que realiza resumenes de alta calidad.user
        Realiza un resumen conciso del siguiente texto: {} Proporciona tu respuesta en
        espanol.assistant"""
    }
    return dict_prompts_template[model_name].format(text)

def get_list_text_recursive_splitter(string_text: str, model_name="mistral"):
    mapper_chunck_size = {
        "mistral": 5000,
    }

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n"],
        chunk_size=mapper_chunck_size[model_name],
        chunk_overlap=mapper_chunck_size[model_name] * 0.05,  # 5% overlapping.
        add_start_index=True,
    )
    return text_splitter.split_text(string_text)


class LLMProcessor:
    def __init__(self, model_name="TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"):
        revision = "gptq-4bit-32g-actorder_True"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=False,
            revision=revision
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.generation_config = GenerationConfig.from_pretrained(model_name)
        self.generation_config.max_new_tokens = 512
        self.generation_config.temperature = 0.7
        self.generation_config.do_sample = True
        self.generation_config.top_k = 40
        self.generation_config.top_p = 0.95
        self.generation_config.repetition_penalty = 1.1
        self.llm = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            generation_config=self.generation_config,
        )
        
    def process_summary(self, text):
        prompt_template = get_template_to_summary_llms_mistral_base(text, model_name="mistral")
        return self.llm(prompt_template)[0]['generated_text'].split("[/INST]")[-1]

    def batch_process(self, texts):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.process_summary, texts))
        return results

if __name__ == '__main__':
    with open("./data/T-273-01.txt", "r") as doc:
        text = doc.read()

    processor = LLMProcessor()
    list_text_splitter = get_list_text_recursive_splitter(text)
    summaries = processor.batch_process(list_text_splitter)
    
    final_summary = ' '.join(summaries)
    print(final_summary)
    print(text)

 
#============== OTRA FROMA DE GENERAR TEXTO  ============================================= #
# print("\n\n*** Generate:")
# prompt = "Tell me about AI"
# prompt_template=f"""<s>[INST] {prompt} [/INST]"""
# input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
# output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
# print(tokenizer.decode(output[0]))
# ======================================= OTRA FROMA DE GENERAR TEXTO  ============================================= #
