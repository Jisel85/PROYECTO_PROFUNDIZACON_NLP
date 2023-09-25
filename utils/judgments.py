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
FOLDER_TO_READ = MAIN_PATH / "data" / "downloaded_judgments"
url_base = "https://www.corteconstitucional.gov.co/relatoria/"


def get_url_judgment(judgment):
    """
    TUTELA:             https://www.corteconstitucional.gov.co/relatoria/1992/T-612-92.htm
    AUTOS:              https://www.corteconstitucional.gov.co/relatoria/autos/1992/A024-92.htm
    CONSTITUCIONAL:     https://www.corteconstitucional.gov.co/relatoria/1992/C-587-92.htm
    :param judgment: id_judgment
    :return: url to try download judgment.
    """
    year = judgment.split("-")[-1]
    year = re.sub(r"[^a-zA-Z0-9]", "", year)
    if float(year) > 91:
        year = "19" + year
    elif float(year) < 24:
        year = "20" + year
    if judgment.lower().startswith("a"):
        url_judgment = f"{url_base}autos/{year}/{judgment}.htm"
        return url_judgment
    else:
        url_judgment = f"{url_base}{year}/{judgment}.htm"
        return url_judgment


def write_json_judgments_urls(folder: pathlib.Path) -> None:
    """
    A JSON file is obtained with the judgment (id_judgment) information (key) and its url (value).
    :param folder: Folder with the Excel files (.xlsx) where the judgments are detailed. folder must be a Path object.
    :return: None
    """
    if not (MAIN_PATH / "data" / "judgments.json").exists():
        with open("data/judgments.json", "w") as file:
            json.dump({}, file)

    for file_excel_relatoria in folder.iterdir():
        df_relatoria = pd.read_excel(file_excel_relatoria, skiprows=6)
        df_relatoria.dropna(subset=["sentenciav2"], inplace=True)
        logging.info(
            f"File: {file_excel_relatoria.name}. With dropna by 'sentenciav2' has shape: {df_relatoria.shape}"
        )
        df_relatoria = df_relatoria.assign(
            url_judgment=df_relatoria["sentenciav2"].apply(get_url_judgment)
        )
        dict_judgments = {
            row["sentenciav2"]: row["url_judgment"]
            for _, row in df_relatoria.iterrows()
        }

        with open("data/judgments.json", "r") as f:
            json_judgments = json.loads(f.read())
            json_judgments.update(dict_judgments)

        with open("data/judgments.json", "w") as f:
            f.write(json.dumps(json_judgments))


def get_text_judgment_raw(judgment):
    """
    :param judgment: judgment's name (Example: "C-010-95")
    :return: String with the text raw in download judgment.
    """
    path_file_judgment = FOLDER_TO_READ / f"{judgment}.rtf"
    try:
        with open(path_file_judgment, "r") as doc:
            rtf_text = doc.read()
        string_judgment = rtf_to_text(rtf_text)
        return string_judgment
    except:
        logging.exception(f"¡Have an exception in judgment: {judgment}!")


class MongoJudgments:
    """
    The init is a mongodb's name of in localhost. Must be existed the collection judgments in the database 
    Project_NLP to use this class. Note: Start connexion with MongoDB in local --->  sudo systemctl start mongod
    """
    mongo_client = MongoClient(host="localhost", port=27017)
    database = mongo_client["Project_NLP"]
    judgment_collection = database["judgments"]

    def __init__(self, judgment: str):
        self.judgment = judgment

    def _verification(self) -> None:
        """
        Create a verification to use Class.
        """
        if "Project_NLP" not in self.mongo_client.list_database_names():
            raise ValueError(
                f"Database Project_NLP does not exist. Must be create the DB in mongo first!"
            )

        if "judgments" not in self.database.list_collection_names():
            raise ValueError(
                f"The collection judgments in MongoDB Project_NLP does not exist. Must be create first!"
            )

    def _write_id_judgments_in_collection(self):
        """
        :return:
        """
        collection = self.judgment_collection
        with open("data/judgments.json", "r") as f:
            json_judgments = json.loads(f.read())
        if self.judgment in json_judgments.keys():
            document = {
                "id_judgment": self.judgment,
            }
            existing_document = collection.find_one(
                {"id_judgment": document["id_judgment"]}
            )
            if existing_document is not None:
                pass
            else:
                collection.insert_one(document)
        else:
            raise ValueError(
                f"The judgment with id_judgment: {self.judgment} doesn't in judgments.json."
            )

    def _write_in_collection_judgments(self, content, feature):
        """
        :param content:
        :param feature:
        :return:
        """
        self._verification()
        self._write_id_judgments_in_collection()
        collection = self.judgment_collection
        collection.update_one(
            filter={
                "id_judgment": self.judgment
            },
            update={
                "$set": {
                    f"{feature}": content
                }
            }
        )

    def write_judgment_feature(self, content: str, feature: str) -> None:
        """
        This function write info in mongodb's collection about judgment. id_judgment will be key identification.
        :param content: String to write.
        :param feature:
        :return:
        """
        # Falta realizar una verificación para evitar que se sobreescriba la información. OJO. #
        self._write_in_collection_judgments(content, feature)

    def get_feature_of_judgment(self, feature):
        """
        :param feature:
        :return:
        """
        collection = self.judgment_collection
        query = {"id_judgment": self.judgment}
        return collection.find_one(query)[feature]
