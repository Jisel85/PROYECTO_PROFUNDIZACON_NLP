import nltk
import time
from nltk.corpus import stopwords
from functions import upload_excel 
from functions import strip_accents 
from functions import limpiar_sintesis
from functions import limpiar_y_aplicar_stopwords
from functions import eliminar_palabras_cortas
from functions import worldcloud_generation
from functions import conteos

def final_execution():
    # Registra el tiempo de inicio
    start_time = time.time()

    sentence_data = upload_excel('archivo_sentencias_1992_2023.xlsx')

    conteos(sentence_data)

    for column in sentence_data.keys():
        sentence_data[column] = sentence_data[column].map(strip_accents)

    # Aplica la función de limpieza a la columna "sintesis"
    sentence_data['sintesis'] = sentence_data['sintesis'].apply(limpiar_sintesis)

    # Descarga las stopwords si aún no las tienes
    nltk.download('stopwords')

    # Supongamos que tienes un DataFrame llamado sentence_data

    # Crea una lista de stopwords en español
    stop_words = set(stopwords.words('spanish'))

    # Aplica la función a la columna "sintesis" y crea la nueva columna "texto_limpio"
    #sentence_data['texto_limpio'] = sentence_data['sintesis'].apply(limpiar_y_aplicar_stopwords(stop_words))
    sentence_data['texto_limpio'] = sentence_data['sintesis'].apply(lambda x: limpiar_y_aplicar_stopwords(x, stop_words))

    # Aplica la función a la columna "texto_limpio" y crea la nueva columna "texto_limpio_2"
    sentence_data['texto_limpio_2'] = sentence_data['texto_limpio'].apply(eliminar_palabras_cortas)

    filtered_data = sentence_data[(sentence_data['sintesis'].notna()) & (sentence_data['sintesis'] != 'Sin informacion')]

    worldcloud_generation('texto_limpio_2',"nube_palabras_1.PNG",filtered_data)

    stop_words_custom = set([
        "accion","tutela","derechos","derecho","fundamentales","sala","plena","fundamental","juzgado","debido","proceso",
        "mismo","luego"
        # Agrega más palabras personalizadas aquí
    ])

    stop_words_combined = stop_words.union(stop_words_custom)

    # Aplica la función a la columna "sintesis" y crea la nueva columna "texto_limpio"
    #filtered_data['texto_limpio_3'] = filtered_data['texto_limpio_2'].apply(limpiar_y_aplicar_stopwords(stop_words_combined))
    filtered_data['texto_limpio_3'] = filtered_data['texto_limpio_2'].apply(lambda x: limpiar_y_aplicar_stopwords(x, stop_words_combined))
    
    worldcloud_generation('texto_limpio_3',"nube_palabras_2.PNG",filtered_data)

    # Registra el tiempo de finalización
    end_time = time.time()

    # Calcula el tiempo transcurrido
    elapsed_time = end_time - start_time

    # Imprime el tiempo transcurrido
    print(f"Tiempo de ejecución: {elapsed_time} segundos")

final_execution()