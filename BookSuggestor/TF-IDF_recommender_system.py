import requests
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "books_with_TF_IDF.csv")

# Cargar base local
def load_database():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df["description"] = df["description"].fillna("")
        print(f" Base cargada con {len(df)} libros.")
        return df
    else:
        print(" No se encontró archivo local. Se usará base vacía.")
        columns = ["title", "author", "categories", "description", "language",
                   "pageCount", "averageRating", "ratingsCount", "publisher",
                   "publishedDate"]
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

# Guardar nuevo libro a base local
def add_to_database(df, book):
    exists = ((df["title"].str.lower() == book["title"].lower()) & 
              (df["author"].str.lower() == book["author"].lower())).any()
    if exists:
        print("El libro ya está en la base.")
        return df

    df = pd.concat([df, pd.DataFrame([book])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")
    return df

# Recomendador basado en TF-IDF
def recommend_books_tfidf(query_title, df, top_n=5, save_new=True):
    print(f"\n Buscando: {query_title}")
    book = search_google_books(query_title)
    if not book or not book["description"]:
        print(" Libro no encontrado o sin descripción.")
        return df

    print(f" Título: {book['title']} \n Autor: {book['author']}\n Descripción: {book['description'][:150]}...\n")

    if save_new:
        df = add_to_database(df, book)

    if len(df) < 2:
        print(" No hay suficientes libros para comparar.")
        return df

    # Preparar corpus
    corpus = df["description"].tolist()
    query_index = len(df) - 1

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    query_vec = tfidf_matrix[query_index]
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    sorted_indices = np.argsort(similarities)[::-1]
    filtered_indices = [i for i in sorted_indices if i != query_index][:top_n]

    print(f"\n Recomendaciones similares (TF-IDF):\n")
    for i in filtered_indices:
        print(f" {df.loc[i, 'title']} — {df.loc[i, 'author']}")
        print(f"    Similitud: {similarities[i]:.4f}\n")

    return df

# MAIN
if __name__ == "__main__":
    base_df = load_database()
    while True:
        query = input("\n Ingresa el título del libro (o 'salir'): ")
        if query.lower() == "salir":
            break
        base_df = recommend_books_tfidf(query, base_df, top_n=5, save_new=True)

    

