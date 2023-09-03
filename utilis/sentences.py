import os
import re
import json
import logging
from pathlib import Path
import pandas as pd

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


sentences_webscraping = MAIN_PATH / "data" / "data_sentences"
get_json_sentences_urls(sentences_webscraping)
