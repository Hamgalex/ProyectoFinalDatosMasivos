import requests
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

# === Configuración de rutas ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "books_with_TF_IDF.csv")

# === Stopwords inglés + español ===
# nltk.download('stopwords')
stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))
stop_words |= {
    # Inglés: términos editoriales y de estructura narrativa
    "book", "books", "story", "stories", "novel", "novels",
    "author", "writer", "writing", "written", "read", "reading", "reader",
    "chapter", "chapters", "page", "pages", "edition", "publishing", "published",

    # Español: términos editoriales y de estructura narrativa
    "libro", "libros", "novela", "novelas", "autor", "autora",
    "leer", "lectura", "leído", "escrito", "escribir",
    "página", "páginas", "capítulo", "capítulos", "edición", "publicado", "editorial",
    "Â", "Ã", "ð", "œ", "â", "€", "™"
}


def fix_encoding(text):
    try:
        return text.encode('latin1').decode('utf-8')
    except:
        return text

# === Cargar base local ===
def load_database():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, encoding="utf-8")
        df["description"] = df["description"].fillna("")
        print(f"Base cargada con {len(df)} libros.")
        return df
    else:
        print("No se encontró archivo local. Se usará base vacía.")
        columns = ["title", "author", "categories", "description", "language",
                   "pageCount", "averageRating", "ratingsCount", "publisher",
                   "publishedDate"]
        return pd.DataFrame(columns=columns)


# === Buscar libro en Google Books ===
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


# === Guardar libro a CSV ===
def add_to_database(df, book):
    exists = ((df["title"].str.lower() == book["title"].lower()) &
              (df["author"].str.lower() == book["author"].lower())).any()
    if exists:
        print("El libro ya está en la base.")
        return df

    df = pd.concat([df, pd.DataFrame([book])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")
    return df


# === Generar nube de palabras ===
def generar_nube(descripciones, nombre_base="wordcloud_tfidf.png"):
    texto = " ".join(descripciones)
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


# === Recomendador TF-IDF con nube ===
def recommend_books_tfidf(query_title, df, top_n=5, save_new=True):
    print(f"\nBuscando: {query_title}")
    book = search_google_books(query_title)
    if not book or not book["description"]:
        print("Libro no encontrado o sin descripción.")
        return df

    print(f"Título: {book['title']}\nAutor: {book['author']}\nDescripción: {book['description'][:150]}...\n")

    if save_new:
        df = add_to_database(df, book)

    if len(df) < 2:
        print("No hay suficientes libros para comparar.")
        return df

    # TF-IDF
    corpus = df["description"].tolist()
    query_index = len(df) - 1
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    query_vec = tfidf_matrix[query_index]
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    sorted_indices = np.argsort(similarities)[::-1]

    # Excluir el documento con similitud más cercana a 1.0 (el mismo)
    filtered_indices = [i for i in sorted_indices if similarities[i] < 0.9999][:top_n]

    print("\nRecomendaciones similares (TF-IDF):\n")
    descripciones_recomendadas = []

    for i in filtered_indices:
        print(f"{df.loc[i, 'title']} — {df.loc[i, 'author']}")
        print(f"Similitud: {similarities[i]:.4f}\n")
        descripcion_limpia = fix_encoding(df.loc[i, "description"])
        descripciones_recomendadas.append(descripcion_limpia)

    generar_nube(descripciones_recomendadas, f"wordcloud_{query_title.lower().replace(' ', '_')}.png")
    return df


# === Main loop ===
if __name__ == "__main__":
    base_df = load_database()
    while True:
        query = input("\nIngresa el título del libro (o 'salir'): ")
        if query.lower() == "salir":
            break
        base_df = recommend_books_tfidf(query, base_df, top_n=5, save_new=True)
