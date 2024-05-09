import sys
import spacy
import logging

spacy.prefer_gpu()

from spacy.lang.es.stop_words import STOP_WORDS  # Importa las palabras de parada para español
from string import punctuation
import string
from spacy.lang.es import Spanish  # Importa la clase de lenguaje para español
from heapq import nlargest
punctuations = string.punctuation
from spacy.language import Language
from utils.judgments import MongoJudgments
from sentence_transformers import SentenceTransformer, util

import pinecone
from getpass import getpass
from pinecone import Pinecone, ServerlessSpec

import logging
from summarization.models_gguf import get_summary_from_gguf_llm
from summarization.models_api import get_summary_from_openai, get_summary_from_huggingface
from summarization.models_gptq import get_summary_from_gptq
from summarization.model_zephyr import get_summary_from_zephyr

import os
import datetime

from utils.judgments import MongoJudgments

nlp = Spanish()
nlp.add_pipe('sentencizer')
parser = Spanish()

from spacy.lang.es.stop_words import STOP_WORDS  # Importa las palabras de parada para español
from string import punctuation

def pre_process(document):
    clean_tokens = [token.lemma_.lower().strip() for token in document]
    clean_tokens = [token for token in clean_tokens if token not in STOP_WORDS and token not in punctuation]

    # La siguiente línea genera una lista de tokens en minúsculas, pero como ya estamos limpiando los tokens anteriormente,
    # puedes considerar eliminarla si no la necesitas para otra cosa.
    lower_case_tokens = [token.text.lower() for token in document]

    return lower_case_tokens

# Generar vectores de números a partir de texto
# Un diccionario con un valor numerico para cada uno de los tokens

def generate_numbers_vector(tokens):
    diccionario_tokens = {}
    for token in tokens:
        diccionario_tokens[token] = diccionario_tokens.get(token, 0) + 1
    frequency = [diccionario_tokens[token] for token in tokens] # complejidad N
    # frequency = [tokens.count(token) for token in tokens] # complejidad N cuadrado
    token_dict = dict(list(zip(tokens,frequency)))
    maximum_frequency=sorted(token_dict.values())[-1]
    normalised_dict = {token_key:token_dict[token_key]/maximum_frequency for token_key in token_dict.keys()}
    return normalised_dict

# Generar puntaje de importancia de oraciones

def sentences_importance(text, normalised_dict):
    importance ={}
    for sentence in nlp(text).sents:
        for token in sentence:
            target_token = token.text.lower()
            if target_token in normalised_dict.keys():
                if sentence in importance.keys():
                    importance[sentence]+=normalised_dict[target_token]
                else:
                    importance[sentence]=normalised_dict[target_token]
    return importance

# Generar resumen

def generate_summary(rank, text):
    target_document = parser(text)
    importance = sentences_importance(text, generate_numbers_vector(pre_process(target_document)))
    summary = nlargest(rank, importance, key=importance.get)
    return summary

def dividir_texto(text):
    longitud = len(text)
    mitad = longitud // 2
    text1 = text[:mitad]
    text2 = text[mitad:]
    return text1, text2

def procesar_texto(text1, text2):
    num_sentences_to_generate = 5
    resumen1 = generate_summary(num_sentences_to_generate, text1)
    resumen2 = generate_summary(num_sentences_to_generate, text2)
    return resumen1, resumen2

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

pc = Pinecone(api_key='683338d7-6863-4d59-9ba5-6f4d422bab9d')
index = pc.Index("relatoria-emebeddings")

from pymongo import MongoClient

mongo_client = MongoClient(host="localhost", port=27017)
database = mongo_client["Project_NLP"]
judgment_collection = database["judgments"]

log_dir = '/home/ladyotavo/nlp/proyecto/logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Obtener la fecha y hora actual
current_time = datetime.datetime.now()

# Formatear la fecha y hora en el formato deseado
log_filename = current_time.strftime('procesamiento_%Y-%m-%d_%H-%M-%S.log')

# Configurar el registro con el nombre de archivo generado
logging.basicConfig(filename=os.path.join(log_dir, log_filename), level=logging.INFO, format='%(asctime)s - %(message)s')

# Iterar sobre los documentos en la colección
for documento in judgment_collection.find():
    providencia_ = documento["id_judgment"]

    logging.info(f"Inicia providencia: {providencia_}")

    print(">>>> Inicia providencia", providencia_)

    index = pc.Index("relatoria-emebeddings")
    query = ''
    query_vector = model.encode(query).tolist()
    responses = index.query(vector=query_vector, top_k = 3, include_metadata=True)

    query_result = index.query(
            vector=query_vector,
            filter={
                "Providencia": {"$eq": providencia_},
            },
            top_k=1,
            include_metadata=True
        )

    # Extraer el ID de la base de datos de Pinecone
    pinecone_id = query_result['matches'][0]['id'] if query_result['matches'] else None

    print(">>>> ID providencia", providencia_, ":", pinecone_id)

    mayor_que = 17635

    try:
        if int(pinecone_id) > mayor_que:
            try:
                client_m = MongoJudgments(providencia_)
                text = client_m.get_feature_of_judgment("raw_text")

                text1, text2 = dividir_texto(text)
                resumen1, resumen2 = procesar_texto(text1, text2)
                resumen_final = str(resumen1+resumen2)

                resultados = index.query(
                    vector=query_vector,
                    filter={
                        "Providencia": {"$eq": providencia_},
                    },
                    top_k=1,
                    include_metadata=True
                )

                # Seleccionar solo el campo "id" de los resultados
                ids = [match['id'] for match in resultados['matches']]

                # Imprimir los ids
                for id in ids:
                    id = id

                index.update(id=id, set_metadata={"summary_extract": resumen_final, "new": "true"})
                
                # Registrar el procesamiento en el archivo de registro
                logging.info(f"Procesamiento extractivo exitoso para providencia: {providencia_}")
            
            except Exception as e:  # Captura cualquier otro tipo de excepción
                logging.error(f"Procesamiento con error en providencia {providencia_}: {str(e)}")
                continue  # Omitir procesamiento para evitar interrupciones
            
            finally:
                print("Termina providencia:", providencia_)
        else:
            print("Providencia con ID menor a", mayor_que, ":", providencia_)

            # Registrar el procesamiento en el archivo de registro
            logging.info(f"Procesamiento ya regsitrado para: {providencia_}")
    except Exception as e:
        print("Error: El ID Pinecone '{}' no es válido.".format(pinecone_id))
        print("Error:", e)
        continue