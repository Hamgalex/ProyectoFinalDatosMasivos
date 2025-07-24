import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from sentence_transformers import util
from wordcloud import WordCloud

# === CONFIGURACIÓN ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "1001_books_with_embeddings.csv")

# Stopwords en inglés y español
stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))
stop_words |= {
    "book", "books", "story", "stories", "novel", "novels",
    "author", "writer", "writing", "written", "read", "reading", "reader",
    "chapter", "chapters", "page", "pages", "edition", "publishing", "published",
    "libro", "libros", "novela", "novelas", "autor", "autora",
    "leer", "lectura", "leído", "escrito", "escribir",
    "página", "páginas", "capítulo", "capítulos", "edición", "publicado", "editorial"
}

# === Función para corregir errores de codificación ===
def fix_encoding(text):
    try:
        return text.encode('latin1').decode('utf-8')
    except:
        return text

def recommend_books(book_title, top_n=5):
    matches = df[df["title"].str.lower() == book_title.lower()]
    if matches.empty:
        print(f"No se encontró el libro: {book_title}")
        return

    idx = matches.index[0]
    query_embedding = embedding_matrix[idx]

    similarities = util.cos_sim(query_embedding, embedding_matrix)[0].cpu().numpy()
    similar_indices = np.argsort(similarities)[::-1][1 : top_n + 1]

    print(f"\nRecomendaciones similares a: **{df.loc[idx, 'title']}**\n")

    # Recolectar descripciones corregidas
    textos_recomendados = []

    for i in similar_indices:
        desc = df.loc[i, 'description']
        if isinstance(desc, float):  # Si es NaN
            desc = ""
        desc = fix_encoding(desc)  # ← Aplica corrección aquí
        print(f"🔹 {df.loc[i, 'title']} — {df.loc[i, 'author']}")
        print(f"     {desc[:150]}...")
        print(f"     Similitud: {similarities[i]:.4f}\n")
        textos_recomendados.append(desc)

    # === NUBE DE PALABRAS ===
    texto_final = " ".join(textos_recomendados)
    wordcloud = WordCloud(
        width=1200,
        height=700,
        background_color="white",
        stopwords=stop_words,
        max_words=150,
        contour_width=1,
        contour_color='steelblue'
    ).generate(texto_final)

    output_name = f"wordcloud_{book_title.lower().replace(' ', '_')}.png"
    output_path = os.path.join(BASE_DIR, output_name)

    plt.figure(figsize=(14, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"🖼️ Nube de palabras guardada en: {output_path}")

if __name__ == "__main__":
    print("Cargando libros con embeddings...")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")  # ← Mejor lectura segura
    df["embedding"] = df["embedding"].apply(eval)
    embedding_matrix = torch.tensor(df["embedding"].tolist())

    query = input("Ingresa el título de un libro: ")
    recommend_books(query, top_n=5)
