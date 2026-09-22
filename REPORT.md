# Multi-modal RAG: Report

## Dataset

14 world landmarks, each with one photo and one short encyclopedic description:
Eiffel Tower, Statue of Liberty, Great Wall of China, Taj Mahal, Colosseum, Big Ben,
Sydney Opera House, Golden Gate Bridge, Machu Picchu, Great Pyramid of Giza, Mount
Fuji, Stonehenge, Burj Khalifa, and Leaning Tower of Pisa.

Images and text were pulled from Wikipedia's public API (`build_dataset.py`): the
`pageimages` endpoint supplies each article's lead photograph, and the
`page/summary` endpoint supplies a short text extract. Wikipedia content is
licensed under CC BY-SA, and images are either public domain or CC-licensed via
Wikimedia Commons. The mapping is stored in `data/metadata.csv` and
`data/metadata.json` with columns `image_path`, `topic`, `text`.

## Architecture

Two embedding spaces are used, matched to what each is good at:

- **CLIP** (`openai/clip-vit-base-patch32`) embeds both images and text into the
  same 512-dimensional space, so it powers the cross-modal search: text-to-image
  and image-to-text.
- **Sentence-transformer** (`all-MiniLM-L6-v2`) embeds only text, but produces
  higher-quality text-to-text semantic similarity than CLIP's text tower, which was
  trained primarily to align with images rather than to compare text against text.

Three FAISS `IndexFlatIP` indices are built over normalized vectors so inner
product is equivalent to cosine similarity:

- `image_clip_index`: CLIP embeddings of all 14 images.
- `text_clip_index`: CLIP embeddings of all 14 facts.
- `text_st_index`: sentence-transformer embeddings of all 14 facts.

Retrieval functions (`vector_store.py`):

- `retrieve_images_for_text(query)` embeds the query with CLIP's text encoder and
  searches `image_clip_index`.
- `retrieve_texts_for_image(image_path)` embeds the image with CLIP's image
  encoder and searches `text_clip_index`.
- `retrieve_texts_for_text(query)` embeds the query with the sentence transformer
  and searches `text_st_index`.

Generation uses `facebook/opt-1.3b` (`generation.py`). Retrieved context is
inserted into a prompt that instructs the model to answer only from that context.

## Example results

```
[text-to-image] Query: a tall bridge over water
  - Golden Gate Bridge (score: 0.2734)
  - Great Wall of China (score: 0.2630)
  - Eiffel Tower (score: 0.2383)
Answer: The Golden Gate Bridge, which is the tallest bridge in the world, is
located in San Francisco, California. It is a steel truss bridge with a span of
1,068 feet (335 meters). The bridge was built in 1937 by the San Francisco
Municipal Railway. The bridge is named

[image-to-text] Query image: data/images/taj_mahal.jpeg
  - Taj Mahal (score: 0.2745)
  - Eiffel Tower (score: 0.2116)
  - Great Pyramid of Giza (score: 0.1779)
Answer: The Taj Mahal is an ivory-white marble mausoleum on the right bank of the
river Yamuna in Agra, Uttar Pradesh, India. It was commissioned in 1631 by the
fifth Mughal emperor, Shah Jahan, to house the tomb of his late wife

[text-to-text] Query: Who built the Eiffel Tower?
  - Eiffel Tower (score: 0.7878)
  - Burj Khalifa (score: 0.4443)
  - Leaning Tower of Pisa (score: 0.3116)
Answer: Gustave Eiffel, the engineer who designed and built the Eiffel Tower.
```

Retrieval correctly identified the right landmark for all three queries,
including the purely descriptive text-to-image query with no landmark name in it.
The text-to-text similarity score (0.79) is much higher than the cross-modal
scores (0.2-0.3), which is expected: CLIP's shared space is optimized for
image-text alignment, not for maximizing absolute similarity scores, so
cross-modal scores sit on a different scale than text-to-text ones and shouldn't
be compared directly across index types.

## Limitations and possible improvements

- OPT-1.3B is a base language model, not instruction-tuned, so its continuations
  sometimes drift past the requested answer into unrelated invented details (the
  Golden Gate Bridge answer above adds a fabricated construction detail). An
  instruction-tuned model would follow the "answer only from context" instruction
  more reliably.
- CLIP's text tower truncates at 77 tokens, so long facts get cut off before
  embedding; a longer-context text encoder would preserve more detail for
  cross-modal matching.
- With only 14 items, there is no meaningful threshold for rejecting a bad match;
  a larger, more diverse dataset would make it possible to set a similarity
  cutoff below which the system reports "no relevant item found" instead of
  returning its best (possibly irrelevant) guess.
- Fusing the two text embedding spaces (CLIP and sentence-transformer) into a
  single ranked result, instead of keeping them as separate indices for separate
  query types, could make results more consistent across query modes.
