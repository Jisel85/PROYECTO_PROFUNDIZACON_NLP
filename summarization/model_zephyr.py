import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from utils.summary import get_template_to_summary_llms_mistral_base


def get_summary_from_zephyr(text, model_name="HuggingFaceH4/zephyr-7b-alpha"):
    """
    Implementation of the HuggingFace zephyr model.
    WARNING: Minimum 16GB GPU required for operation (Free Google colab)
    Vea https://docs.mistral.ai/quickstart
    :param text:
    :param model_name:
    :return:
    """
    generation_config = GenerationConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

  feature/testing_llms
    generation_config.max_new_tokens = 512
    generation_config.temperature = 1
    generation_config.do_sample = False
    generation_config.top_k = 80
=======
    generation_config.max_new_tokens = 256
    generation_config.temperature = 0.7
    generation_config.do_sample = True
    generation_config.top_k = 50
 main
    generation_config.top_p = 0.95

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
 feature/testing_llms
        return_full_text=False,
=======
        return_full_text=True,
 main
    )

    prompt_template = get_template_to_summary_llms_mistral_base(text, model_name="zephyr")

    return pipe(prompt_template)
