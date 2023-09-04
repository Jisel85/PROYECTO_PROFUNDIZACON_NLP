import os
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
url_base = "https://www.corteconstitucional.gov.co/relatoria/"


def get_url_sentence(sentence):
    """
    TUTELA:             https://www.corteconstitucional.gov.co/relatoria/1992/T-612-92.htm
    AUTOS:              https://www.corteconstitucional.gov.co/relatoria/autos/1992/A024-92.htm
    CONSTITUCIONAL:     https://www.corteconstitucional.gov.co/relatoria/1992/C-587-92.htm
    :param sentence:
    :return:
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


def get_json_sentences_urls(folder):
    """
    Obtenemos archivo JSON con la información de sentencia (key) y su url (value).
    :param folder: Folder con los archivos excel (.xlsx) donde se detallan las sentencias.
    :return:
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
    path_to_read = MAIN_PATH / "data" / "sentences_webscraping" / f"{sentence}.rtf"
    with open(path_to_read, "r") as doc:
        rtf_text = doc.read()
    string_sentence = rtf_to_text(rtf_text)
    return string_sentence


class Sentences:
    """
    Note: Start connexion with MongoDB in local --->  sudo systemctl start mongod
    """

    def __init__(self, db_name: str):
        self.db_name = db_name
        self.mongo_client = MongoClient(host="localhost", port=27017)
        self.database = None
        self.db_collections = None
        self.MAIN_PATH = Path(os.getcwd())
        self.PATH_TO_READ = self.MAIN_PATH / "data" / "downloads_files_sentences"
        self.PATH_TO_WRITE = self.MAIN_PATH / "data" / "preprocessing_corpus"

    def _get_connection_database_mongo(self):
        """
        Create a connection to the MongoDB DB from the init.
        :return: A database object
        """
        if self.db_name not in self.mongo_client.list_database_names():
            raise ValueError(f"Database {self.db_name} does not exist. Must be create the DB in mongo first!")
        self.database = self.mongo_client[self.db_name]
        return self.database

    def get_collection(self, name_collection):
        """
        :param name_collection: Name of collection in MongoDB
        :return: pymongo.collection.Collection
        """
        database = self._get_connection_database_mongo()
        self.db_collections = self.database.list_collection_names()
        if name_collection not in self.db_collections:
            raise ValueError(f"The collection {name_collection} in MongoDB {self.db_name} does not exist.")
        collection = database[name_collection]
        return collection

    def write_sentence_in_mongodb(self, sentence, name_collection: str):
        """
        This function write in mongoDB the sentence_raw (Downloaded in .rft)
        :param name_collection:
        :param sentence:
        :return:
        """
        sentence_raw = get_text_sentence_raw(sentence)
        document = {"id_sentence": sentence, "text_raw": sentence_raw}
        # try:
        existing_document = self.get_collection(name_collection).find_one({"id_sentence": document["id_sentence"]})
        if existing_document is not None:
            raise ValueError(f"No es posible reescribir la información de la sentencia.!")

        self.get_collection(name_collection).insert_one(document)
        self.mongo_client.close()
        # except:
        #     print(f"The database {self.db_name} has a new collection: {name_collection} OJO")
        #     collection = self.database[name_collection]
        #     collection.insert_one(document)
