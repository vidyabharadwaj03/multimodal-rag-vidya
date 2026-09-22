import csv

from generation import (
    build_image_to_text_prompt,
    build_text_to_image_prompt,
    build_text_to_text_prompt,
    generate,
)
from vector_store import MultiModalIndex

METADATA_PATH = "data/metadata.csv"


def load_records():
    with open(METADATA_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def print_retrieved(retrieved):
    for record, score in retrieved:
        print(f"  - {record['topic']} (score: {score:.4f})")


def run_text_to_image(index, query):
    print(f"[text-to-image] Query: {query}")
    retrieved = index.retrieve_images_for_text(query, k=3)
    print_retrieved(retrieved)
    answer = generate(build_text_to_image_prompt(query, retrieved))
    print(f"Answer: {answer}\n")


def run_image_to_text(index, image_path):
    print(f"[image-to-text] Query image: {image_path}")
    retrieved = index.retrieve_texts_for_image(image_path, k=3)
    print_retrieved(retrieved)
    answer = generate(build_image_to_text_prompt(retrieved))
    print(f"Answer: {answer}\n")


def run_text_to_text(index, query):
    print(f"[text-to-text] Query: {query}")
    retrieved = index.retrieve_texts_for_text(query, k=3)
    print_retrieved(retrieved)
    answer = generate(build_text_to_text_prompt(query, retrieved))
    print(f"Answer: {answer}\n")


def main():
    records = load_records()
    index = MultiModalIndex(records)
    index.build()

    run_text_to_image(index, "a tall bridge over water")
    run_image_to_text(index, "data/images/taj_mahal.jpeg")
    run_text_to_text(index, "Who built the Eiffel Tower?")


if __name__ == "__main__":
    main()
