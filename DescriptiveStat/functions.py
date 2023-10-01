import pandas as pd
import os
import unicodedata
import re
import numpy as np
import string  # Importa la biblioteca de signos de puntuación
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def upload_excel(file_name):
    # Obtiene el directorio de trabajo actual
    current_directory = os.getcwd()
    subcarpeta="DescriptiveStat"

    # Define el nombre del archivo
    archivo_nombre = file_name

    # Combina el directorio actual con el nombre del archivo
    archivo_ruta = os.path.join(current_directory, subcarpeta, archivo_nombre)
    
    # Carga el archivo Excel en un DataFrame
    sentence_data = pd.read_excel(archivo_ruta)

    return sentence_data

def strip_accents(s):
  if (type(s) == str):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
  return s

def limpiar_sintesis(texto):
    if isinstance(texto, str):
        # Elimina texto que inicie con "t-"
        texto = re.sub(r'^t-\S+\s*', '', texto)
        # Elimina números, "/", y guiones "-"
        texto = re.sub(r'[0-9/\\-]+', '', texto)
        
        # Elimina signos de puntuación
        texto = texto.translate(str.maketrans('', '', string.punctuation))
        
        return texto
    else:
        return texto
    
def limpiar_y_aplicar_stopwords(texto,stop_w):
    if isinstance(texto, str):
        # Tokeniza el texto en palabras
        words = nltk.word_tokenize(texto)
        # Filtra las palabras para eliminar las stopwords
        filtered_words = [word.lower() for word in words if word.lower() not in stop_w]
        # Convierte las palabras en un solo texto
        texto_limpio = " ".join(filtered_words)
        return texto_limpio
    else:
        return texto

# Define una función para eliminar palabras de tres o menos caracteres
def eliminar_palabras_cortas(texto):
    if isinstance(texto, str):
        palabras = texto.split()  # Divide el texto en palabras
        palabras_filtradas = [palabra for palabra in palabras if len(palabra) > 3]
        return ' '.join(palabras_filtradas)
    else:
        return texto
    
def worldcloud_generation(select_column,wcl_name,data_name):
    # Concatena todas las entradas de la columna "texto_limpio_2" en un solo texto
    texto_limpio_text = ' '.join(data_name[select_column])
    
    # Crea la nube de palabras
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto_limpio_text)

    # Visualiza la nube de palabras
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nube de Palabras')
    
    # Guarda la imagen de la nube de palabras en el directorio actual con el nombre "nube_palabras_1.png"
    nombre_archivo_imagen = wcl_name
    ruta_completa_imagen = os.path.join(os.getcwd(), nombre_archivo_imagen)
    plt.savefig(ruta_completa_imagen)

def contar_nulos(x):
    return (x.isna() | (x == 'Sin informacion')).sum()

def conteos(entrada):
    conteo_tipos = entrada['Tipo'].value_counts()

    # Utiliza groupby() en la columna "Tipo" y aplica la función personalizada
    conteo_valores_nulos_por_tipo = entrada.groupby('Tipo')['sintesis'].apply(contar_nulos)

    # Combina los dos DataFrames para obtener el resultado final.
    resultado = pd.merge(conteo_tipos, conteo_valores_nulos_por_tipo, on='Tipo')
    # Crea la columna "porcentaje" calculando "sintesis" dividido por "count"
    resultado['porcentaje'] = resultado['sintesis'] / resultado['count'] * 100
    # Redondea la columna "porcentaje" a dos decimales
    resultado['porcentaje'] = resultado['porcentaje'].round(2)

    return resultado.to_excel('conteos.xlsx', index=False)