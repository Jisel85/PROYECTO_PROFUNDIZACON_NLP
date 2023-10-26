from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from utils.summary import get_template_to_summary_llms_mistral_base


class GPTQProcessor:
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
        print("Llamado a summary!")
        prompt_template = get_template_to_summary_llms_mistral_base(text, model_name="mistral")
        return self.llm(prompt_template)[0]['generated_text'].split("[/INST]")[-1]

    def batch_process(self, texts):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.process_summary, texts))
        return results
