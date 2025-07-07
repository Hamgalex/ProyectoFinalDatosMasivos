import pandas as pd
from sentence_transformers import SentenceTransformer
import ast

# Cargar CSV original
df = pd.read_csv("../book_downloader/1001_books_detailed.csv")
df["description"] = df["description"].fillna("")

# Modelo
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Generar embeddings
print("🔄 Generando embeddings...")
embeddings = model.encode(df["description"].tolist(), convert_to_numpy=True)

# Guardar como listas normales de float (no np.float32)
df["embedding"] = [list(map(float, emb)) for emb in embeddings]

# Guardar a CSV
df.to_csv("1001_books_with_embeddings.csv", index=False)
print("✅ Guardado CSV con embeddings limpios.")



