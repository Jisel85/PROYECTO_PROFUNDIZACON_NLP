import json
import logging

import pandas as pd
from pymongo import MongoClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

mongo_client = MongoClient(host="localhost", port=27017)


def get_dataframe_from_collection_judgments():
    """
    :return:
    """
    collection_judgments = mongo_client["Project_NLP"]["judgments"]
    return pd.DataFrame(collection_judgments.find({}))


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
        :param str feature:
        :return:
        """
        collection = self.judgment_collection
        query = {"id_judgment": self.judgment}
        return collection.find_one(query)[feature]
