from transformers import pipeline

GENERATION_MODEL_NAME = "facebook/opt-1.3b"

_generator = None


def get_generator():
    global _generator
    if _generator is None:
        _generator = pipeline("text-generation", model=GENERATION_MODEL_NAME)
    return _generator


def build_text_to_image_prompt(query, retrieved):
    topics = ", ".join(record["topic"] for record, _ in retrieved)
    return (
        f"Question: {query}\n"
        f"Retrieved images show: {topics}.\n"
        "Answer, using only the retrieved images above, which one best matches the question and why:"
    )


def build_image_to_text_prompt(retrieved):
    facts = "\n".join(f"- {record['text']}" for record, _ in retrieved)
    return (
        "The following facts were retrieved for an image query:\n"
        f"{facts}\n"
        "Summarize what the image most likely depicts, using only the facts above:"
    )


def build_text_to_text_prompt(query, retrieved):
    fact, _ = retrieved[0]
    return (
        f"Fact: {fact['text']}\n"
        f"Question: {query}\n"
        "Answer the question using only the fact above. "
        "If the fact does not contain the answer, say so explicitly instead of guessing.\n"
        "Answer:"
    )


def generate(prompt, max_new_tokens=60):
    generator = get_generator()
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_return_sequences=1,
    )
    generated_text = output[0]["generated_text"]
    return generated_text[len(prompt):].strip()
