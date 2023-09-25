import os
import json
import logging
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from utils.judgments import write_json_judgments_urls

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

MAIN_PATH = Path(os.getcwd())
download_folder_judgments = MAIN_PATH / "data" / "downloaded_judgments"

options = Options()
options.set_preference(name="browser.download.folderList", value=2)
options.set_preference("browser.download.dir", str(download_folder_judgments))
options.add_argument("--headless")

if not (MAIN_PATH / "data" / "judgments.json").exists():
    path_excels = Path("/home/andres-campos/github_private/PROYECTO_PROFUNDIZACON_NLP/data/data_judgments")
    write_json_judgments_urls(path_excels)

with open("data/judgments.json", "r") as file:
    json_judgments = json.loads(file.read())


def open_browser():
    """
    :return: 
    """
    return webdriver.Firefox(options=options)


def get_judgment_file(judgment):
    """
    :param judgment: Label judgment. Ex:
    :return:
    """
    file_path = download_folder_judgments / f"{judgment}.rtf"
    if file_path.exists():
        return None

    url_judgment = json_judgments[judgment]
    driver = open_browser()
    driver.get(url_judgment)
    try:
        radicado_bt = driver.find_element(
            By.XPATH, "/html/body/div[5]/div/div[3]/a/img"
        )
        radicado_bt.click()
    except:
        logging.exception(f"Have a exception in URL: {url_judgment}")
    driver.close()
    return None


judgments = list(json_judgments.keys())
for judgment in judgments:
    get_judgment_file(judgment)
