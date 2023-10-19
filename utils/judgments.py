import os
import json
import logging
import pandas as pd
from pymongo import MongoClient
from pymongo.server_api import ServerApi

ATLAS_CREDENTIAL = os.environ["ATLAS_CREDENTIAL"]


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_dataframe_from_host_judgments(host="local"):
    """
    :return:
    """
    if host == "local":
        mongo_client = MongoClient(host="localhost", port=27017)
        collection_judgments = mongo_client["Project_NLP"]["judgments"]
        return pd.DataFrame(collection_judgments.find({}))
    elif host == "atlas":
        mongo_client_atlas = MongoClient(host=ATLAS_CREDENTIAL, server_api=ServerApi("1"))
        collection_judgments_atlas = mongo_client_atlas["Project_NLP_Atlas"]["judgments_summary"]
        return pd.DataFrame(collection_judgments_atlas.find({}))


class MongoJudgments:
    """
    The init is a id_judgment (Ex: A-100-97). Must be existed the collection judgments in the database
    Project_NLP (local) and judgments_summary in the database Project_NPL_Atlas to use this class.

    Note: (1) Must be passed your credentials to connexion in the global variable "atlas_host"
    """
    mongo_client = MongoClient(host="localhost", port=27017)
    mongo_client_atlas = MongoClient(host=ATLAS_CREDENTIAL, server_api=ServerApi("1"))
    database = mongo_client["Project_NLP"]
    database_atlas = mongo_client_atlas["Project_NLP_Atlas"]
    judgment_collection = database["judgments"]
    judgment_collection_atlas = database_atlas["judgments_summary"]

    def __init__(self, judgment: str):
        self.judgment = judgment

    def _verification(self) -> None:
        """
        Create a verification to use Class.
        """
        if "Project_NLP" not in self.mongo_client.list_database_names() \
                or "Project_NLP_Atlas" not in self.mongo_client_atlas.list_database_names():
            raise ValueError(
                f"Database Project_NLP or Project_NLP_Atlas don't exist. Must be create the DB in mongo first!"
            )

        if "judgments" not in self.database.list_collection_names() \
                or "judgments_summary" not in self.database_atlas.list_collection_names():
            raise ValueError(
                f"The collection judgments in MongoDB Project_NLP does n't exist, or the collection judgment_summary "
                f"doesn't exist in Project_NLP_Atlas. Must be create first!"
            )

    def _selector_mongodb(self, host="local"):
        """
        :param host: Host of MongoDB.
        :return: collection
        """
        collection = self.judgment_collection
        if host == "atlas":
            collection = self.judgment_collection_atlas

        return collection

    def _write_id_judgments_in_collection(self, host="local"):
        """
        :return:
        """
        collection = self._selector_mongodb(host)
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

    def _write_in_collection_judgments(self, content, feature, host):
        """
        :param content:
        :param feature:
        :return:
        """
        self._verification()
        self._write_id_judgments_in_collection(host)
        collection = self._selector_mongodb(host)
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

    def write_judgment_feature(self, content: str, feature: str, host="local") -> None:
        """
        This function write info in mongodb's collection about judgment. id_judgment will be key identification.
        :param content: String to write.
        :param feature: Feature of judgment to get as a string.
        :param host:
        :return:
        """
        # Falta realizar una verificación para evitar que se sobreescriba la información. OJO. #
        self._write_in_collection_judgments(content, feature, host)

    def get_feature_of_judgment(self, feature, host="local"):
        """
        :param str feature:
        :param str host:
        :return:
        """
        collection = self._selector_mongodb(host)
        query = {"id_judgment": self.judgment}

        return collection.find_one(query)[feature]
