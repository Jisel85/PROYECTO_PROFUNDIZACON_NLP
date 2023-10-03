import os
import json
import logging
import concurrent.futures
from pathlib import Path
from preprocessCorte.preprocessing_judgments import write_json_judgments_urls, write_excel_judgments_in_mongo, \
    write_judgment_raw_text

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

MAIN_PATH = Path(os.getcwd())
path_excels = MAIN_PATH / "data" / "data_judgments"

logging.info(f"START THE PROCESS: WRITE IN MONGODB judgments")

write_json_judgments_urls(path_excels)
write_excel_judgments_in_mongo()

with open(MAIN_PATH / "data" / "judgments.json", "r") as f:
    json_judgments = json.loads(f.read())
list_judgments = list(json_judgments.keys())

num_threads = os.cpu_count()

with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    results = list(executor.map(write_judgment_raw_text, list_judgments))

logging.info(f"END THE PROCESS: WRITE IN MONGODB judgments")
