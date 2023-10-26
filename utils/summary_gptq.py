import gc
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

# Si se necesita la función exllama_set_max_input_length, la importamos aquí
try:
    from auto_gptq import exllama_set_max_input_length
except ImportError:
    exllama_set_max_input_length = None

def count_tokens_from_gguf_models(text: str) -> int:
    """Returns the number of tokens in a text string with mistral embeddings."""
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1",
        # token="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"
    )
    return len(tokenizer.encode(text))


def get_template_to_summary_llms_mistral_base(text, model_name="mistral"):
    """
    Obtenemos el prompt para ingresar como input a los modelos basados en Mistral-7b
    :param text:
    :param model_name:
    :return:
    """
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


models_gptq = [
    "TheBloke/dolphin-2.1-mistral-7B-GPTQ""",
    "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",
    "TheBloke/Mistral-7B-OpenOrca-GPTQ",
    "TheBloke/samantha-1.2-mistral-7B-GPTQ",
    "TheBloke/SlimOpenOrca-Mistral-7B-GPTQ",
    "TheBloke/zephyr-7B-alpha-GPTQ",
]


def get_simple_summary(text, model_name="TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"):
    """
    :param text:
    :param model_name:
    :return:
    """
    revision = "gptq-4bit-32g-actorder_True"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=False,
        revision=revision
    )

    if exllama_set_max_input_length:
        model = exllama_set_max_input_length(model, 4096)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    generation_config = GenerationConfig.from_pretrained(model_name)
    generation_config.max_new_tokens = 512
    generation_config.temperature = 0.7
    generation_config.do_sample = True
    generation_config.top_k = 40
    generation_config.top_p = 0.95
    generation_config.repetition_penalty = 1.1

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
    )
    prompt_template = get_template_to_summary_llms_mistral_base(text, model_name="mistral")
    summary_llm = pipe(prompt_template)[0]['generated_text'].split("[/INST]")[-1]
    return summary_llm


def get_list_text_recursive_splitter(string_text: str, model_name="mistral"):
    """
    Gets the text of the context size according to the structure that will be used for the summary.
    :return:
    """
    mapper_chunck_size = {
        "mistral": 5000,
    }

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n"],
        chunk_size=mapper_chunck_size[model_name],
        chunk_overlap=mapper_chunck_size[model_name] * 0.05,  # 5% overlapping.
        add_start_index=True,
    )
    list_text_splitter = text_splitter.split_text(string_text)

    return list_text_splitter


def get_map_summary(text, model_name="TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"):
    """
    Esta funcion retorna resumen por tramos del documento, es decir, se parte en chuncks el documento y se generan
    resumem de cada chunck. Retorna una lista con el resumen de cada chunck. Los resumenes siguen el orden de la
    ocurrecia en el documento.
    Models options: mistral-7b-openorca, zephyr-7b-alpha, mistral-7b-instruct-v0.1.
    :param text: Recibe el texto raw como string.
    :param model_name:
    :return:
    """
    revision = "gptq-4bit-32g-actorder_True"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=False,
        revision=revision
    )
    
    if exllama_set_max_input_length:
        model = exllama_set_max_input_length(model, 4096)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    generation_config = GenerationConfig.from_pretrained(model_name)
    generation_config.max_new_tokens = 512
    generation_config.temperature = 0.7
    generation_config.do_sample = True
    generation_config.top_k = 40
    generation_config.top_p = 0.95
    generation_config.repetition_penalty = 1.1

    llm = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
    )

    list_text_splitter, map_summary = get_list_text_recursive_splitter(text), []

    for text_splitter in list_text_splitter:
        prompt_template = get_template_to_summary_llms_mistral_base(text_splitter, model_name="mistral")
        print(text_splitter)
        summary_llm = llm(prompt_template)[0]['generated_text'].split("[/INST]")[-1]
        print("Summary:", summary_llm)
        map_summary.append(summary_llm)

    del model, llm, tokenizer, generation_config
    gc.collect()
    time.sleep(5)
    return map_summary


def user_summary(text: str, model_name="TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"):
    """
    :param text:
    :param model_name:
    :return:
    """
    if count_tokens_from_gguf_models(text) < 1200:
        simple_summary = get_simple_summary(text)
        return simple_summary

    map_summary = get_map_summary(text, model_name=model_name)
    doc = "\n\n".join(map_summary)
    print(doc)
    print("Tokens:", count_tokens_from_gguf_models(doc))
    #  Falta solucionar si el mapeo superar los 1200 tokens!!!!
    # revision = "gptq-4bit-32g-actorder_True"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=False,
        revision="main"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    generation_config = GenerationConfig.from_pretrained(model_name)
    generation_config.max_new_tokens = 1024
    generation_config.temperature = 0.7
    generation_config.do_sample = True
    generation_config.top_k = 40
    generation_config.top_p = 0.95
    generation_config.repetition_penalty = 1.1

    llm = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
    )

    prompt_template = get_template_to_summary_llms_mistral_base(text, model_name="mistral")
    summary_llm = llm(prompt_template)[0]['generated_text'].split("[/INST]")[-1]
    return summary_llm


with open("./data/T-273-01.txt", "r") as doc:
    text = doc.read()

summary = user_summary(text)
print(summary)
print(text)




# ======================================= OTRA FROMA DE GENERAR TEXTO  ============================================= #
# print("\n\n*** Generate:")
# prompt = "Tell me about AI"
# prompt_template=f"""<s>[INST] {prompt} [/INST]"""
# input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
# output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
# print(tokenizer.decode(output[0]))
# ======================================= OTRA FROMA DE GENERAR TEXTO  ============================================= #
