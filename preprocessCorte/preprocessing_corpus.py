from utilis.sentences import Sentences, get_json_sentences_urls
from pathlib import Path

# Example use get_json_sentences_urls. Getting sentences.json file.
path_excels = Path("/home/andres-campos/github_private/PROYECTO_PROFUNDIZACON_NLP/data/data_sentences")
get_json_sentences_urls(path_excels)

#  Example use class Sentences.
test = Sentences("sentences")
test.write_sentence_in_mongodb(sentence="C-010-95", name_collection="sentences_raw")

