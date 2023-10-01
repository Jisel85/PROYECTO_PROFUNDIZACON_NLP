import json
import logging
import concurrent.futures
from functools import partial
from pathlib import Path
from utilis.sentences import Sentences, get_json_sentences_urls

import sys
sys.path.append('d:\\Angela\\Maestría\\Trabajo_Grado\\PROYECTO_PROFUNDIZACON_NLP')

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info(f"START THE PROCESS: WRITE IN MONGODB SENTENCES")

# Example use get_json_sentences_urls. Getting sentences.json file.
path_excels = Path("d:\\Angela\\Maestría\\Trabajo_Grado\\PROYECTO_PROFUNDIZACON_NLP\\data\\data_judgments")
get_json_sentences_urls(path_excels)

with open("data/sentences.json", "r") as f:
    json_sentences = json.loads(f.read())
list_sentences = list(json_sentences.keys())

# Example use class Sentences.
test = Sentences(db_name="sentences")
write_to_collection_urls = partial(test.write_info_sentence_in_collection, name_collection="urls")
write_to_collection_texts = partial(test.write_info_sentence_in_collection, name_collection="raw_texts")


def write_to_collections(sentence):
    """
    :param sentence:
    :return:
    """
    write_to_collection_urls(sentence)
    write_to_collection_texts(sentence)


max_threads = 8
with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
    executor.map(write_to_collections, list_sentences)
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


num_threads = 4

with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    results = list(executor.map(process_judgment, list_judgments))

logging.info(f"END THE PROCESS: WRITE IN MONGODB SENTENCES")
