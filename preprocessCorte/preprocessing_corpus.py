import json
import logging
import concurrent.futures
from functools import partial
from pathlib import Path
from utils.sentences import Sentences, get_json_sentences_urls

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info(f"START THE PROCESS: WRITE IN MONGODB SENTENCES")

# Example use get_json_sentences_urls. Getting sentences.json file.
path_excels = Path("/home/andres-campos/github_private/PROYECTO_PROFUNDIZACON_NLP/data/data_sentences")
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

logging.info(f"END THE PROCESS: WRITE IN MONGODB SENTENCES")
