import os
import json
import re
import logging
import pandas as pd
from pymongo import MongoClient
from pathlib import Path
from striprtf.striprtf import rtf_to_text
from utils.judgments import MongoJudgments

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info(f"START THE PROCESS: WRITE IN MONGODB judgments")

MAIN_PATH = Path(os.getcwd())
FOLDER_TO_READ = MAIN_PATH / "data" / "downloaded_judgments"
url_base = "https://www.corteconstitucional.gov.co/relatoria/"


def get_url_download_judgment(judgment):
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


def write_json_judgments_urls(folder: Path) -> None:
    """
    A JSON file is obtained with the judgment (id_judgment) information (key) and its url (value).
    :param folder: Folder with the Excel files (.xlsx) where the judgments are detailed. folder must be a Path object.
    :return: None
    """
    if (MAIN_PATH / "data" / "judgments.json").exists():
        pass
    else:
        with open("data/judgments.json", "w") as file:
            json.dump({}, file)

        for file_excel_relatoria in folder.iterdir():
            df_judgment = pd.read_excel(file_excel_relatoria)
            logging.info(
                f"File: {file_excel_relatoria.name}. Has shape: {df_judgment.shape}"
            )
            df_judgment = df_judgment.assign(
                url_judgment=df_judgment["Providencia"].apply(get_url_download_judgment)
            )
            dict_judgments = {
                row["Providencia"]: row["url_judgment"]
                for _, row in df_judgment.iterrows()
            }

            with open(MAIN_PATH / "data" / "judgments.json", "r") as f:
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


def write_excel_judgments_in_mongo():
    """
    :return:
    """
    judgment_collection = MongoClient(host="localhost", port=27017)["Project_NLP"]["judgments"]

    df_judgment = pd.read_excel("data/data_judgments/archivo sentencias 1992-2023.xlsx", index_col=0)
    df_judgment.columns = [column.replace(" - ", "_").replace(" ", "_").lower() for column in df_judgment.columns]
    df_judgment.rename(columns={"providencia": "id_judgment"}, inplace=True)
    df_judgment["f_public"].fillna("N/A", inplace=True)
    df_judgment["fecha_sentencia"].fillna("N/A", inplace=True)
    df_judgment = df_judgment.assign(
        url_download=df_judgment["id_judgment"].apply(get_url_download_judgment)
    )

    records_dict = df_judgment.to_dict(orient="records")

    for record in records_dict:
        try:
            judgment_collection.insert_one(record)
        except Exception as e:
            logging.exception(f"{e}")


def write_judgment_raw_text(judgment):
    """
    :param judgment: id_judgment
    :return:
    """
    try:
        judgment_client = MongoJudgments(judgment=judgment)
        judgment_client.write_judgment_feature(content=get_text_judgment_raw(judgment), feature="raw_text")
    except Exception as e:
        logging.exception(f"Judgment {judgment}\n2 {e}")
