import csv
import json
import time
import urllib.parse
from pathlib import Path

import requests

USER_AGENT = "multimodal-rag-lab/1.0 (educational project)"
DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
METADATA_PATH = DATA_DIR / "metadata.csv"

TOPICS = [
    "Eiffel Tower",
    "Statue of Liberty",
    "Great Wall of China",
    "Taj Mahal",
    "Colosseum",
    "Big Ben",
    "Sydney Opera House",
    "Golden Gate Bridge",
    "Christ the Redeemer",
    "Machu Picchu",
    "Great Pyramid of Giza",
    "Mount Fuji",
    "Stonehenge",
    "Burj Khalifa",
    "Leaning Tower of Pisa",
]


def get_with_retry(url, params=None, retries=5, backoff=5):
    for attempt in range(retries):
        response = requests.get(
            url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30
        )
        if response.status_code == 429:
            time.sleep(backoff * (attempt + 1))
            continue
        response.raise_for_status()
        return response
    response.raise_for_status()
    return response


def fetch_thumbnail_url(title):
    params = {
        "action": "query",
        "titles": title,
        "prop": "pageimages",
        "format": "json",
        "pithumbsize": 800,
    }
    response = get_with_retry("https://en.wikipedia.org/w/api.php", params=params)
    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("thumbnail", {}).get("source")


def fetch_summary(title):
    encoded_title = urllib.parse.quote(title.replace(" ", "_"))
    response = get_with_retry(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
    )
    return response.json()["extract"]


def download_image(url, destination):
    response = get_with_retry(url)
    destination.write_bytes(response.content)


def build():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for topic in TOPICS:
        existing = list(IMAGES_DIR.glob(topic.lower().replace(" ", "_") + ".*"))
        if existing:
            text = fetch_summary(topic)
            rows.append({"image_path": str(existing[0]), "topic": topic, "text": text})
            time.sleep(1.5)
            continue

        thumbnail_url = fetch_thumbnail_url(topic)
        if not thumbnail_url:
            continue

        text = fetch_summary(topic)
        extension = Path(urllib.parse.urlparse(thumbnail_url).path).suffix or ".jpg"
        filename = topic.lower().replace(" ", "_") + extension
        image_path = IMAGES_DIR / filename

        download_image(thumbnail_url, image_path)
        rows.append({"image_path": str(image_path), "topic": topic, "text": text})
        time.sleep(1.5)

    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "topic", "text"])
        writer.writeheader()
        writer.writerows(rows)

    with open(DATA_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved {len(rows)} image/text pairs to {DATA_DIR}")


if __name__ == "__main__":
    build()
