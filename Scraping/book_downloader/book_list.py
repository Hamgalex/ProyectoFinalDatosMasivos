import requests
from bs4 import BeautifulSoup
import csv
import re
import ftfy
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URL = "https://notesfromabooknerd.com/reading-list-challenges/1001-books-you-must-read-before-you-die/"

def get_books():
    response = requests.get(URL)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    content_div = soup.find("div", class_="entry-content")
    ol = content_div.find("ol")

    if not ol:
        raise ValueError("No se encontró la lista <ol> en el contenido.")

    items = ol.find_all("li")
    print(f" Se encontraron {len(items)} libros.")

    books = []
    for item in items:
        full_text = ftfy.fix_text(item.get_text(strip=True))
        fixed_text = re.sub(r'\s*by\s*', ' by ', full_text, flags=re.IGNORECASE)

        if " by " in fixed_text:
            title, author = fixed_text.rsplit(" by ", 1)
        else:
            title, author = fixed_text, ""

        books.append({
            "title": title.strip(),
            "author": author.strip()
        })

    return books

def save_to_csv(books, filename="1001_books_list.csv", folder_path = BASE_DIR):
        # Si se pasa una ruta, la combina con el nombre del archivo
    if folder_path:
        os.makedirs(folder_path, exist_ok=True)
        filename = os.path.join(folder_path, filename)
        
    with open(filename, mode="w", encoding="utf-8-sig", newline="") as f:  #  utf-8-sig para Excel
        writer = csv.DictWriter(f, fieldnames=["title", "author"])
        writer.writeheader()
        writer.writerows(books)
    print(f"CSV guardado como: {filename}")

if __name__ == "__main__":
    books = get_books()
    save_to_csv(books)
