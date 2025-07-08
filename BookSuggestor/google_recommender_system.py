import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import ast
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_FILE = os.path.join(BASE_DIR, "books_with_embeddings.csv")

# Cargar modelo
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Cargar base local
def load_embedding_database():
    if os.path.exists(EMBEDDING_FILE):
        df = pd.read_csv(EMBEDDING_FILE)
        df["embedding"] = df["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
        print(f" Base cargada con {len(df)} libros.")
        return df
    else:
        print(" No se encontró archivo local. Se usará base vacía.")
        columns = ["title", "author", "categories", "description", "language",
                   "pageCount", "averageRating", "ratingsCount", "publisher",
                   "publishedDate", "embedding"]
        return pd.DataFrame(columns=columns)

# Buscar libro en Google Books
def search_google_books(title):
    params = {"q": title, "maxResults": 1}
    response = requests.get("https://www.googleapis.com/books/v1/volumes", params=params)
    data = response.json()

    if "items" not in data:
        return None

    info = data["items"][0]["volumeInfo"]
    return {
        "title": info.get("title", ""),
        "author": ", ".join(info.get("authors", [])),
        "categories": ", ".join(info.get("categories", [])),
        "description": info.get("description", ""),
        "language": info.get("language", ""),
        "pageCount": info.get("pageCount", ""),
        "averageRating": info.get("averageRating", ""),
        "ratingsCount": info.get("ratingsCount", ""),
        "publisher": info.get("publisher", ""),
        "publishedDate": info.get("publishedDate", "")
    }


# Generar embedding de descripción
def generate_embedding(text):
    return model.encode(text, convert_to_numpy=True)

# Guardar nuevo libro a base local
def add_to_database(df, book, emb):
    # Evitar agregar si ya existe (por título y autor)
    exists = ((df["title"].str.lower() == book["title"].lower()) & 
              (df["author"].str.lower() == book["author"].lower())).any()
    if exists:
        print("El libro ya está en la base.")
        return df

    new_row = book.copy()
    new_row["embedding"] = emb.tolist()

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(EMBEDDING_FILE, index=False, encoding="utf-8-sig")
    print("Nuevo libro agregado a la base.")
    return df


# Recomendador principal
def recommend_books(query_title, df, top_n=5, save_new=True):
    print(f"\n Buscando: {query_title}")
    book = search_google_books(query_title)
    if not book or not book["description"]:
        print(" Libro no encontrado o sin descripción.")
        return

    print(f" Título: {book['title']} \n Autor: {book['author']}\n Descripción: {book['description'][:150]}...\n")
    emb = generate_embedding(book["description"])

    if save_new:
        df = add_to_database(df, book, emb)

    if len(df) == 0:
        print(" No hay libros en la base para comparar.")
        return

    # Comparar
    emb_matrix = np.vstack(df["embedding"].values).astype(np.float32)

    query_emb = torch.tensor(emb, dtype=torch.float32)
    embedding_tensor = torch.tensor(emb_matrix, dtype=torch.float32)

    similarities = util.cos_sim(query_emb, embedding_tensor)[0].cpu().numpy()
    sorted_indices = np.argsort(similarities)[::-1]

    query_index = len(df) - 1  # índice del libro recién agregado

    filtered_indices = [i for i in sorted_indices if i != query_index][:top_n]

    print(f"\n Recomendaciones similares:\n")
    for i in filtered_indices:
        print(f" {df.loc[i, 'title']} — {df.loc[i, 'author']}")
        print(f"    Similitud: {similarities[i]:.4f}\n")

    return df

# MAIN
if __name__ == "__main__":
    base_df = load_embedding_database()
    while True:
        query = input("\n Ingresa el título del libro (o 'salir'): ")
        if query.lower() == "salir":
            break
        base_df = recommend_books(query, base_df, top_n=5, save_new=True)
    

