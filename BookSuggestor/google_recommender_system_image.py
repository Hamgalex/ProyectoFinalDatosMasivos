import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import ast
import os
import torch
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_FILE = os.path.join(BASE_DIR, "books_with_embeddings.csv")

# === Modelo ===
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# === Stopwords básicas (solo meta-literarias)
stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))
stop_words |= {
    "book", "books", "story", "stories", "novel", "novels",
    "author", "writer", "writing", "written", "read", "reading", "reader",
    "chapter", "chapters", "page", "pages", "edition", "publishing", "published",
    "libro", "libros", "novela", "novelas", "autor", "autora",
    "leer", "lectura", "leído", "escrito", "escribir",
    "página", "páginas", "capítulo", "capítulos", "edición", "publicado", "editorial"
}

# === Corregir encoding mal leído ===
def fix_encoding(text):
    try:
        return text.encode('latin1').decode('utf-8')
    except:
        return text

# === Cargar base local ===
def load_embedding_database():
    if os.path.exists(EMBEDDING_FILE):
        df = pd.read_csv(EMBEDDING_FILE)
        df["embedding"] = df["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
        print(f"Base cargada con {len(df)} libros.")
        return df
    else:
        print("No se encontró archivo local. Se usará base vacía.")
        columns = ["title", "author", "categories", "description", "language",
                   "pageCount", "averageRating", "ratingsCount", "publisher",
                   "publishedDate", "embedding"]
        return pd.DataFrame(columns=columns)

# === Buscar en Google Books ===
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

# === Generar embedding ===
def generate_embedding(text):
    return model.encode(text, convert_to_numpy=True)

# === Guardar nuevo libro ===
def add_to_database(df, book, emb):
    exists = ((df["title"].str.lower() == book["title"].lower()) &
              (df["author"].str.lower() == book["author"].lower())).any()
    if exists:
        print("El libro ya está en la base.")
        return df

    new_row = book.copy()
    new_row["embedding"] = emb.tolist()

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df["embedding"] = df["embedding"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    df.to_csv(EMBEDDING_FILE, index=False, encoding="utf-8-sig")
    return df

# === Generar nube de palabras ===
def generar_nube(descripciones, nombre_base="wordcloud.png"):
    texto = " ".join(fix_encoding(desc) for desc in descripciones)
    wordcloud = WordCloud(
        width=1200,
        height=700,
        background_color="white",
        stopwords=stop_words,
        max_words=150,
        contour_width=1,
        contour_color='steelblue'
    ).generate(texto)

    ruta = os.path.join(BASE_DIR, nombre_base)
    plt.figure(figsize=(14, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(ruta)
    print(f"Nube de palabras guardada en: {ruta}")

# === Recomendador ===
def recommend_books(query_title, df, top_n=5, save_new=True):
    print(f"\nBuscando: {query_title}")
    book = search_google_books(query_title)
    if not book or not book["description"]:
        print("Libro no encontrado o sin descripción.")
        return df

    print(f"Título: {book['title']} \nAutor: {book['author']}\nDescripción: {book['description'][:150]}...\n")
    emb = generate_embedding(book["description"])

    if save_new:
        df = add_to_database(df, book, emb)

    if len(df) == 0:
        print("No hay libros en la base para comparar.")
        return df

    emb_matrix = np.vstack(df["embedding"].values).astype(np.float32)
    query_emb = torch.tensor(emb, dtype=torch.float32)
    embedding_tensor = torch.tensor(emb_matrix, dtype=torch.float32)

    similarities = util.cos_sim(torch.tensor(query_emb).unsqueeze(0), embedding_tensor)[0].cpu().numpy()

    # Excluir la posición con similitud exactamente 1.0 (el propio libro)
    sorted_indices = np.argsort(similarities)[::-1]
    filtered_indices = [i for i in sorted_indices if similarities[i] < 0.9999][:top_n]

    print("\nRecomendaciones similares:\n")
    descripciones_recomendadas = []

    for i in filtered_indices:
        print(f"{df.loc[i, 'title']} — {df.loc[i, 'author']}")
        print(f"Similitud: {similarities[i]:.4f}\n")
        descripciones_recomendadas.append(df.loc[i, "description"])

    generar_nube(descripciones_recomendadas, f"wordcloud_{query_title.lower().replace(' ', '_')}.png")
    return df

# === MAIN ===
if __name__ == "__main__":
    base_df = load_embedding_database()
    while True:
        query = input("\nIngresa el título del libro (o 'salir'): ")
        if query.lower() == "salir":
            break
        base_df = recommend_books(query, base_df, top_n=5, save_new=True)
