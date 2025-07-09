import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool, cpu_count

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "../book_downloader/1001_books_detailed.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "1001_books_with_embeddings.csv")

"""
Map: recibe una lista de (índice, descripción) y devuelve una lista de (índice, embedding)
"""
def encode_descriptions(pairs):
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return [(idx, model.encode(desc, convert_to_numpy=True)) for idx, desc in pairs]

"""
Reduce: toma todos los resultados de los procesos y los junta en un diccionario {indice: embedding}
"""
def reduce_results(mapped_data):
    result = {}
    for chunk in mapped_data:
        for idx, emb in chunk:
            result[idx] = list(map(float, emb))
    return result

"""
divide la lista total de datos en n partes iguales (o casi)
"""
def chunkify(data, n):
    k = len(data) // n
    return [data[i * k : (i + 1) * k] if i != n - 1 else data[i * k :] for i in range(n)]

if __name__ == "__main__":
    print("Cargando datos de libros")
    df = pd.read_csv(INPUT_FILE)
    df["description"] = df["description"].fillna("")

    book_pairs = list(zip(df.index, df["description"]))
    chunks = chunkify(book_pairs, cpu_count())

    print("Generando embeddings en paralelo aplicando map")
    with Pool(processes=cpu_count()) as pool:
        mapped = pool.map(encode_descriptions, chunks)

    print("Aplicando reduce")
    result_dict = reduce_results(mapped)

    df["embedding"] = df.index.map(lambda idx: result_dict.get(idx))
    df.to_csv(OUTPUT_FILE, index=False)
    print("Embeddings guardados en:", OUTPUT_FILE)
