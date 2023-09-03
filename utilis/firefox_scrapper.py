import os
import json
import logging
import concurrent.futures
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

MAIN_PATH = Path(os.getcwd())
download_folder_sentences = MAIN_PATH / "data" / "sentences_webscraping"

options = Options()
options.set_preference(name="browser.download.folderList", value=2)
options.set_preference("browser.download.dir", str(download_folder_sentences))
options.add_argument("--headless")

with open("data/sentences.json", "r") as file:
    json_sentences = json.loads(file.read())


def open_browser():
    """"""
    return webdriver.Firefox(options=options)


def get_sentence_by_url(sentence):
    """"""
    file_path = download_folder_sentences / f"{sentence}.rtf"
    if file_path.exists():
        return None

    url_sentence = json_sentences[sentence]
    driver = open_browser()
    driver.get(url_sentence)
    try:
        radicado_bt = driver.find_element(
            By.XPATH, "/html/body/div[5]/div/div[3]/a/img"
        )
        radicado_bt.click()
    except:
        logging.exception(f"Have a exception in URL: {url}")
    driver.close()
    return None


urls = list(json_sentences.keys())
for url in urls:
    get_sentence_by_url(url)
