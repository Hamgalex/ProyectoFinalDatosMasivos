import requests
import csv
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "1001_books_list.csv")
OUTPUT_FILE = os.path.join(BASE_DIR,"1001_books_detailed.csv")

API_URL = "https://www.googleapis.com/books/v1/volumes"
API_KEY = None  # Reemplaza con tu API key si tienes una

def search_book(title, author):
    query = f"{title} {author}"
    params = {
        "q": query,
        "maxResults": 1,
    }
    if API_KEY:
        params["key"] = API_KEY

    response = requests.get(API_URL, params=params)
    if response.status_code != 200:
        return None

    data = response.json()
    if "items" not in data:
        return None

    return data["items"][0]["volumeInfo"]

def enrich_books():
    enriched = []

    with open(INPUT_FILE, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            title, author = row["title"], row["author"]
            print(f" [{i}] Buscando: {title} by {author}")

            info = search_book(title, author)
            time.sleep(0.1)

            if info:
                enriched.append({
                    "title": title,
                    "author": author,
                    "categories": ", ".join(info.get("categories", [])),
                    "description": info.get("description", "").replace("\n", " ").strip(),
                    "language": info.get("language", ""),
                    "pageCount": info.get("pageCount", ""),
                    "averageRating": info.get("averageRating", ""),
                    "ratingsCount": info.get("ratingsCount", ""),
                    "publisher": info.get("publisher", ""),
                    "publishedDate": info.get("publishedDate", "")
                })
            else:
                print(f"No se encontró info para: {title}")
                enriched.append({
                    "title": title,
                    "author": author,
                    "categories": "",
                    "description": "",
                    "language": "",
                    "pageCount": "",
                    "averageRating": "",
                    "ratingsCount": "",
                    "publisher": "",
                    "publishedDate": ""
                })

    return enriched

def save_to_csv(data, filename=OUTPUT_FILE):
    with open(filename, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"\nCSV guardado como: {filename}")

if __name__ == "__main__":
    books_detailed = enrich_books()
    save_to_csv(books_detailed)
