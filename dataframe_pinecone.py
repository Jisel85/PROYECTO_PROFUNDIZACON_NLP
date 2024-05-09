from pinecone import Pinecone

pc = Pinecone(api_key='683338d7-6863-4d59-9ba5-6f4d422bab9d')
index = pc.Index("relatoria-emebeddings")

import time
import concurrent.futures
import pandas as pd

inicio = 0 # o empezar de 0 para otra cosa
fin = 41712
tamaño_chunk = 1000

df = pd.DataFrame()

chunks = []
for i in range(inicio, fin + 1, tamaño_chunk):
    chunks.append((i, min(fin, i + tamaño_chunk - 1)))

def process(row):
    return row['metadata']

for inicio, fin in chunks:
    print('iniciando', inicio, fin)
    resultados = index.fetch(ids=list(map(str, range(inicio, fin))))
    print('datos obtenidos', inicio, fin)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        lista = executor.map(process, list(resultados['vectors'].values()))
        otro_df = pd.DataFrame(lista)
        df = pd.concat([df, otro_df])
    print('datos procesados', inicio, fin)

