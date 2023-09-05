import os
import pathlib
import re
import json
import logging
from pathlib import Path
import pandas as pd
from pymongo import MongoClient
from striprtf.striprtf import rtf_to_text

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

MAIN_PATH = Path(os.getcwd())
FOLDER_TO_READ = MAIN_PATH / "data" / "sentences_webscraping"
url_base = "https://www.corteconstitucional.gov.co/relatoria/"


def get_url_sentence(sentence):
    """
    TUTELA:             https://www.corteconstitucional.gov.co/relatoria/1992/T-612-92.htm
    AUTOS:              https://www.corteconstitucional.gov.co/relatoria/autos/1992/A024-92.htm
    CONSTITUCIONAL:     https://www.corteconstitucional.gov.co/relatoria/1992/C-587-92.htm
    :param sentence:
    :return: url to try download sentence.
    """
    year = sentence.split("-")[-1]
    year = re.sub(r"[^a-zA-Z0-9]", "", year)
    if float(year) > 91:
        year = "19" + year
    elif float(year) < 24:
        year = "20" + year
    if sentence.lower().startswith("a"):
        url_sentence = f"{url_base}autos/{year}/{sentence}.htm"
        return url_sentence
    else:
        url_sentence = f"{url_base}{year}/{sentence}.htm"
        return url_sentence


def get_json_sentences_urls(folder: pathlib.Path) -> None:
    """
    A JSON file is obtained with the sentence (id_sentence) information (key) and its url (value).
    :param folder: Folder with the Excel files (.xlsx) where the sentences are detailed. folder must be a Path object.
    :return: None
    """
    if not (MAIN_PATH / "data" / "sentences.json").exists():
        with open("data/sentences.json", "w") as file:
            json.dump({}, file)

    for file_excel_relatoria in folder.iterdir():
        df_relatoria = pd.read_excel(file_excel_relatoria, skiprows=6)
        df_relatoria.dropna(subset=["sentenciav2"], inplace=True)
        logging.info(
            f"File: {file_excel_relatoria.name}. With dropna by 'sentenciav2' has shape: {df_relatoria.shape}"
        )
        df_relatoria = df_relatoria.assign(
            url_sentence=df_relatoria["sentenciav2"].apply(get_url_sentence)
        )
        dict_sentences = {
            row["sentenciav2"]: row["url_sentence"]
            for _, row in df_relatoria.iterrows()
        }

        with open("data/sentences.json", "r") as f:
            json_sentences = json.loads(f.read())
            json_sentences.update(dict_sentences)

        with open("data/sentences.json", "w") as f:
            f.write(json.dumps(json_sentences))


def get_text_sentence_raw(sentence):
    """
    :param sentence: Sentence's name (Example: "C-010-95")
    :return: String with the text raw in download sentence.
    """
    path_file_sentence = FOLDER_TO_READ / f"{sentence}.rtf"
    try:
        with open(path_file_sentence, "r") as doc:
            rtf_text = doc.read()
        string_sentence = rtf_to_text(rtf_text)
        return string_sentence
    except:
        logging.exception(f"¡Have an exception in sentence: {sentence}!")


class Sentences:
    """
    The init is a mongodb's name of  in localhost. Must be existed the collections raw_texts and urls to use this class.

    Note: Start connexion with MongoDB in local --->  sudo systemctl start mongod
    """

    def __init__(self, db_name: str):
        self.db_name = db_name
        self.mongo_client = MongoClient(host="localhost", port=27017)
        self.database = None

    def _get_connection_mongodb(self) -> None:
        """
        Create a connection to the MongoDB DB from the init.
        """
        if self.db_name not in self.mongo_client.list_database_names():
            raise ValueError(
                f"Database {self.db_name} does not exist. Must be create the DB in mongo first!"
            )
        self.database = self.mongo_client[self.db_name]

    def _get_collection(self, name_collection: str):
        """
        :param name_collection: Name of collection in MongoDB
        :return: pymongo.collection.Collection (name_collection)
        """
        self._get_connection_mongodb()
        if name_collection not in self.database.list_collection_names():
            raise ValueError(
                f"The collection {name_collection} in MongoDB {self.db_name} does not exist."
            )
        return self.database[name_collection]

    def _write_in_collection_raw_texts(self, sentence):
        """
        :param sentence:
        :return:
        """
        collection = self._get_collection("raw_texts")
        text_sentence_raw = get_text_sentence_raw(sentence)
        document = {"id_sentence": sentence, "text_raw": text_sentence_raw}
        existing_document = collection.find_one(
            {"id_sentence": document["id_sentence"]}
        )
        if existing_document is not None:
            raise ValueError(
                f"The sentence with id_sentence: {sentence} already exits in collection."
            )
        collection.insert_one(document)

    def _write_in_collection_urls(self, sentence):
        """
        :param sentence:
        :return:
        """
        collection = self._get_collection("urls")
        with open("data/sentences.json", "r") as f:
            json_sentences = json.loads(f.read())
        url_sentence = json_sentences[sentence]
        document = {"id_sentence": sentence, "url": url_sentence}
        existing_document = collection.find_one(
            {"id_sentence": document["id_sentence"]}
        )
        if existing_document is not None:
            raise ValueError(
                f"The sentence with id_sentence: {sentence} already exits in collection."
            )
        collection.insert_one(document)

    def write_info_sentence_in_collection(self, sentence: str, name_collection: str) -> None:
        """
        This function write info in mongodb's collection about sentence. id_sentence will be key identificator in all
        collections.
        :param sentence: Name of sentence to write.
        :param name_collection: Collection where the information will be written.
        :return:
        """
        if name_collection == "raw_texts":
            self._write_in_collection_raw_texts(sentence)

        elif name_collection == "urls":
            self._write_in_collection_urls(sentence)
# IDEA #
# except:
#     print(f"The database {self.db_name} has a new collection: {name_collection} OJO")
#     collection = self.database[name_collection]
#     collection.insert_one(document)
