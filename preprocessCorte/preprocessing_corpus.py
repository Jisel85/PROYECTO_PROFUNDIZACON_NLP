import json
import logging
import concurrent.futures
from pathlib import Path
from utils.judgments import MongoJudgments, get_text_judgment_raw, write_json_judgments_urls

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info(f"START THE PROCESS: WRITE IN MONGODB judgments")

# Example use get_json_sentences_urls. Getting judgments.json file.
path_excels = Path("/home/andres-campos/github_private/PROYECTO_PROFUNDIZACON_NLP/data/data_sentences")
write_json_judgments_urls(path_excels)

with open("data/judgments.json", "r") as f:
    json_judgments = json.loads(f.read())
list_judgments = list(json_judgments.keys())


def process_judgment(judgment):
    try:
        test = MongoJudgments(judgment=judgment)
        test.write_judgment_feature(content=json_judgments[judgment], feature="url")
        test.write_judgment_feature(content=get_text_judgment_raw(judgment), feature="raw_text")
    except Exception as e:
        logging.exception(f"Judgment {judgment}\n2 {e}")


num_threads = 8

with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    results = list(executor.map(process_judgment, list_judgments))

logging.info(f"END THE PROCESS: WRITE IN MONGODB judgments")
