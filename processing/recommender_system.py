import pandas as pd
import numpy as np
from sentence_transformers import util
import ast

# Cargar el DataFrame con embeddings
csv_path = "1001_books_with_embeddings.csv"
df = pd.read_csv(csv_path)
df["description"] = df["description"].fillna("")

# Convertir los strings de listas a vectores NumPy
df["embedding"] = df["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))

# Crear la matriz de embeddings
embedding_matrix = np.vstack(df["embedding"].values)

# Función de recomendación con similitud
def recommend_books(book_title, top_n=5):
    matches = df[df["title"].str.lower() == book_title.lower()]
    if matches.empty:
        print(f"❌ No se encontró el libro: {book_title}")
        return

    idx = matches.index[0]
    query_embedding = embedding_matrix[idx]

    # Calcular similitudes de coseno
    similarities = util.cos_sim(query_embedding, embedding_matrix)[0].cpu().numpy()
    similar_indices = np.argsort(similarities)[::-1][1 : top_n + 1]

    print(f"\n📚 Recomendaciones similares a: {df.loc[idx, 'title']}\n")
    for i in similar_indices:
        print(f"🔸 {df.loc[i, 'title']} — {df.loc[i, 'author']}")
        print(f"    📖 {df.loc[i, 'description'][:150]}...")
        print(f"    🧠 Similitud: {similarities[i]:.4f}\n")

# Ejecutar si es el script principal
if __name__ == "__main__":

    print("¿Todos los embeddings son iguales?")
    print(np.allclose(embedding_matrix[0], embedding_matrix[1]))  # Compara los dos primeros

