import os
from pathlib import Path
from striprtf.striprtf import rtf_to_text
from pymongo import MongoClient


class Preprocessing:
    ##
    MAIN_PATH = Path(os.getcwd())
    PATH_TO_READ = MAIN_PATH / "data" / "downloads_files_sentences"
    PATH_TO_WRITE = MAIN_PATH / "data" / "preprocessing_corpus"

    def _clean_sentence(self, sentence):
        """"""
        sentence_file = self.PATH_TO_READ / f'{sentence}.rtf'
        with open(sentence_file, 'r') as doc:
            rtf_text = doc.read()
        sentence_raw = rtf_to_text(rtf_text)
        sentence_clean = f"ACA DEBEMOS PONER LA LIMPIEZA"
        return sentence_clean

    def _write_sentence_in_mongodb(self, sentence):
        """
        """
        client = MongoClient('localhost', 27017)
        db = client['sentences']
        collection = db['sentences_raw']
        clean_sentence = self._clean_sentence(sentence)
        document = {'id_sentence': sentence, 'text_raw': clean_sentence, }
        collection.insert_one(document)
        client.close()
