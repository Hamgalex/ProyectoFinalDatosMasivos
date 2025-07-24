import pandas as pd
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

# Asegura que estén descargadas
nltk.download("stopwords")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "../book_downloader/1001_books_detailed.csv")
OUTPUT_IMAGE = os.path.join(BASE_DIR, "wordcloud_books.png")

# Combina stopwords de inglés y español
stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))

# Puedes agregar más palabras irrelevantes si quieres:
stop_words |= {"book", "story", "novel", "life", "one", "man", "woman", "author"}

if __name__ == "__main__":
    print("Cargando datos de libros")
    df = pd.read_csv(INPUT_FILE)
    df["description"] = df["description"].fillna("")

    print("Uniendo todas las descripciones...")
    all_text = " ".join(df["description"].tolist())

    print("Generando la nube de palabras...")
    wordcloud = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        max_words=200,
        stopwords=stop_words,
        collocations=True,
        contour_width=1,
        contour_color='steelblue'
    ).generate(all_text)

    print("Guardando imagen...")
    plt.figure(figsize=(16, 9))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(OUTPUT_IMAGE)
    print("Imagen guardada en:", OUTPUT_IMAGE)
