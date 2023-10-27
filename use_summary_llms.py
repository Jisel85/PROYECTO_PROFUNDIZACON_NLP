import logging
from utils.judgments import MongoJudgments
from summarization.models_gpt4all import get_summary_from_gpt4all
from summarization.models_api import get_summary_from_openai, get_summary_from_huggingface
from summarization.models_gptq import get_summary_from_gptq
from summarization.model_zephyr import get_summary_from_zephyr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info(f"Start the process with LLM local")

# ========================================== Example with gpt4all ================================================= #
judgment = MongoJudgments("T-273-01")  # (Short Judgment, 7 pages)
raw_text = judgment.get_feature_of_judgment("raw_text")
summary = get_summary_from_gpt4all(
    text=raw_text,
    model_name="TheBloke/dolphin-2.1-mistral-7B-GGUF",
    filename="dolphin-2.1-mistral-7b.Q4_0.gguf"
)
logging.info(f"Final summary:\n{summary}")
# ================================================================================================================== #

# # ==========================  Example with openai (Requirement OPENAI_KEY) ========================================= #
judgment = MongoJudgments("T-006-10")  # (Long Judgment, 27 pages)
raw_text = judgment.get_feature_of_judgment("raw_text")
summary = get_summary_from_openai(raw_text)
logging.info(f"Final summary:\n{summary}")
# # ================================================================================================================== #
#
# # =================================== Example with Huggingface API inference ======================================= #
judgment = MongoJudgments("T-991-10")  # (Medium Judgment, 15 pages)
raw_text = judgment.get_feature_of_judgment("raw_text")
summary = get_summary_from_huggingface(raw_text)
logging.info(f"Final summary:\n{summary}")
# # ================================================================================================================== #

# ============================================== Example with GPTQ models ========================================== #
judgment = MongoJudgments("T-273-01")  # (Short Judgment, 7 pages)
raw_text = judgment.get_feature_of_judgment("raw_text")
summary = get_summary_from_gptq(raw_text)
logging.info(f"Final summary:\n{summary}")
# ================================================================================================================== #

# ==============================  Example with LLM based in Mistral local ========================================== #
judgment = MongoJudgments("T-273-01")  # (Short Judgment, 7 pages)
raw_text = judgment.get_feature_of_judgment("raw_text")
summary = get_summary_from_zephyr(raw_text)
logging.info(f"Final summary:\n{summary}")
# ================================================================================================================== #

logging.info(f"Start the process with LLM local")
