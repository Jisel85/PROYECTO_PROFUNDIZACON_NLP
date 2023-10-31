import logging
from utils.judgments import MongoJudgments
from summarization.models_gguf import get_summary_from_gguf_llm
from summarization.models_api import get_summary_from_openai, get_summary_from_huggingface
from summarization.models_gptq import get_summary_from_gptq
from summarization.model_zephyr import get_summary_from_zephyr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info(f"Start the process with LLM's")

# ========================================== Example with gpt4all ================================================= #
judgment = MongoJudgments("T-273-01")  # (Short Judgment, 7 pages)
raw_text = judgment.get_feature_of_judgment("raw_text")
summary = get_summary_from_gguf_llm(
    text=raw_text,
    repo_id="TheBloke/zephyr-7B-beta-GGUF",
    filename="zephyr-7b-beta.Q4_K_M.gguf"
)
logging.info(f"Final summary:\n{summary}")
# ================================================================================================================== #

# ==========================  Example with openai (Requirement OPENAI_KEY) ========================================= #
judgment = MongoJudgments("T-006-10")  # (Long Judgment, 27 pages)
raw_text = judgment.get_feature_of_judgment("raw_text")
summary = get_summary_from_openai(raw_text)
logging.info(f"Final summary:\n{summary}")
# ================================================================================================================== #

# =================================== Example with Huggingface API inference ======================================= #
judgment = MongoJudgments("T-991-10")  # (Medium Judgment, 15 pages)
raw_text = judgment.get_feature_of_judgment("raw_text")
summary = get_summary_from_huggingface(raw_text)
logging.info(f"Final summary:\n{summary}")
# ================================================================================================================== #

# ============================================== Example with GPTQ models ========================================== #
judgment = MongoJudgments("T-273-01")  # (Short Judgment, 7 pages)
raw_text = judgment.get_feature_of_judgment("raw_text")
summary = get_summary_from_gptq(
    raw_text,
    model_name="TheBloke/zephyr-7B-beta-GPTQ",
    revision="gptq-4bit-32g-actorder_True"
)

logging.info(f"Final summary:\n{summary}")
# ================================================================================================================== #

# ==============================  Example with LLM based in Mistral local ========================================== #
judgment = MongoJudgments("T-273-01")  # (Short Judgment, 7 pages)
raw_text = judgment.get_feature_of_judgment("raw_text")
summary = get_summary_from_zephyr(raw_text)
logging.info(f"Final summary:\n{summary}")
# ================================================================================================================== #

logging.info(f"End the process with LLM's")
