import pandas as pd
from sentence_transformers import SentenceTransformer
import ast

# Cargar el CSV original
df = pd.read_csv("../book_downloader/1001_books_detailed.csv")

# Llenar descripciones faltantes
df["description"] = df["description"].fillna("")

# Lista de descripciones
descriptions = df["description"].tolist()

# Cargar modelo BERT multilingüe
print("Cargando modelo")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Generar embeddings
print("🔄 Generando embeddings...")
embeddings = model.encode(descriptions)

# Convertir cada embedding (array) a una lista para guardar en CSV
df["embedding"] = [emb.tolist() for emb in embeddings]

# Guardar CSV con embeddings
output_file = "1001_books_with_embeddings.csv"
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"✅ Embeddings guardados en {output_file}")