import pandas as pd
import numpy as np
from sentence_transformers import util
import ast

# Cargar el DataFrame con embeddings
df = pd.read_csv("1001_books_with_embeddings.csv")
df["description"] = df["description"].fillna("")

# Convertir los embeddings de string a array
df["embedding"] = df["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))

# Crear matriz de embeddings
embedding_matrix = np.vstack(df["embedding"].values)

# Función de recomendación
def recommend_books(book_title, top_n=5):
    matches = df[df["title"].str.lower() == book_title.lower()]
    if matches.empty:
        print(f"❌ No se encontró el libro: {book_title}")
        return

    idx = matches.index[0]
    query_embedding = embedding_matrix[idx]

    similarities = util.cos_sim(query_embedding, embedding_matrix)[0].cpu().numpy()
    similar_indices = np.argsort(similarities)[::-1][1 : top_n + 1]

    print(f"\n📚 Recomendaciones similares a: {df.loc[idx, 'title']}\n")
    for i in similar_indices:
        desc = df.loc[i, 'description']
        if isinstance(desc, float):
            desc = ""
        print(f"🔸 {df.loc[i, 'title']} — {df.loc[i, 'author']}")
        print(f"    📖 {desc[:150]}...")
        print(f"    🧠 Similitud: {similarities[i]:.4f}\n")

# Ejecutar si se llama directamente
if __name__ == "__main__":
    query = input("🔎 Ingresa el título de un libro: ")
    recommend_books(query, top_n=5)
