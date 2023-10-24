import logging
from utils.judgments import MongoJudgments
from utils.summarization import (
    get_summary_from_openai,
    get_summary_from_llm_gpt4all,
    get_summary_from_huggingface_api,
    get_summary_from_llm_mistral_base,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info(f"Start the process with LLM local")

GPU_NAME = "NVIDIA GeForce RTX 3060 Laptop GPU"
# ========================== Example with openai (Requirement OPENAI_KEY) ========================================= #
judgment = MongoJudgments("T-006-10")  # (Long Judgment, 27 pages)
raw_text = judgment.get_feature_of_judgment("raw_text")
summary = get_summary_from_openai(raw_text)
print(summary)
# ================================================================================================================= #
#
# ========================================== Example with gpt4all ================================================= #
judgment = MongoJudgments("T-273-01")  # (Short Judgment, 7 pages)
raw_text = judgment.get_feature_of_judgment("raw_text")
summary = get_summary_from_llm_gpt4all(raw_text, model_name="llama-2-7b-chat", device=GPU_NAME)
print(summary)
# ================================================================================================================= #
#
# =================================== Example with Huggingface API inference ======================================= #
judgment = MongoJudgments("T-991-10")  # (Medium Judgment, 15 pages)
raw_text = judgment.get_feature_of_judgment("raw_text")
print(raw_text)
summary = get_summary_from_huggingface_api(raw_text)
print(summary)
# ================================================================================================================== #

# ==============================  Example with LLM based in Mistral local ========================================== #
judgment = MongoJudgments("T-273-01")  # (Short Judgment, 7 pages)
raw_text = judgment.get_feature_of_judgment("raw_text")
print(raw_text)
summary = get_summary_from_llm_mistral_base(raw_text)
print(summary)
# ================================================================================================================== #

logging.info(f"Start the process with LLM local")