import logging
from utils.judgments import MongoJudgments
from summarization.model_zephyr import get_summary_from_zephyr
import torch

torch.cuda.empty_cache()
judgment = MongoJudgments("T-273-01")  # (Short Judgment, 7 pages)
raw_text = judgment.get_feature_of_judgment("raw_text")
summary = get_summary_from_zephyr(raw_text)
logging.info(f"Final summary:\n{summary}")
