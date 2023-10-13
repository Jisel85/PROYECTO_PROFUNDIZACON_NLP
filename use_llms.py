import logging
from utils.summarization import get_summarization_from_openai

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

summary = get_summarization_from_openai(judgment="T-006-10")
print(summary)

