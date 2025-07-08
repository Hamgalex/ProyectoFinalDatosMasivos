import pandas as pd
from sentence_transformers import SentenceTransformer
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR_2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Subir dos niveles (si estás en processing)
INPUT_FILE = os.path.join(BASE_DIR_2, "book_downloader", "1001_books_detailed.csv")
OUTPUT_FILE_EMBEDDINGS = os.path.join(BASE_DIR,"1001_books_with_embeddings.csv")

# Cargar CSV original
df = pd.read_csv(INPUT_FILE)
df["description"] = df["description"].fillna("")

# Modelo
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Generar embeddings
print(" Generando embeddings...")
embeddings = model.encode(df["description"].tolist(), convert_to_numpy=True)

# Guardar como listas normales de float (no np.float32)
df["embedding"] = [list(map(float, emb)) for emb in embeddings]

# Guardar a CSV
df.to_csv(OUTPUT_FILE_EMBEDDINGS, index=False)
print(" Guardado CSV con embeddings limpios.")



