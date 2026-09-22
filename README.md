# Lab: Multi-modal Retrieval-Augmented Generation (RAG) System

> **Environment reminder:** Use a separate Python 3.11 environment for this lab.
> From this directory, run:
>
> ```bash
> python3.11 -m venv .venv
> source .venv/bin/activate
> python -m pip install --upgrade pip
> ```

---

## Table of Contents

* [Scenario / Background](#scenario--background)
* [Objectives](#objectives)
* [Task Overview](#task-overview)
* [Requirements](#requirements)
* [Data Guidance](#data-guidance)
* [Deliverables](#deliverables)
* [Steps & Recommendations](#steps--recommendations)
* [Extension / Stretch Goals](#extension--stretch-goals)
* [References & Resources](#references--resources)

---

### Scenario / Background

Retrieval-augmented generation (RAG) systems combine language models with an external knowledge store: instead of relying solely on the model's parameters, they retrieve relevant information and then generate a response. In a **multi-modal RAG** system, the knowledge store contains both text and images. Recent advances such as **CLIP** map images and text into the same latent space, enabling cross-modal similarity search. [How CLIP and FAISS work](https://medium.com/@heyitssandeep/how-i-built-an-image-search-engine-with-clip-and-faiss-5b48df1df0fa#:~:text=What%20exactly%20is%20CLIP%20and,FAISS) **FAISS** is an efficient vector index for storing and retrieving dense embeddings, and sentence-transformer models like *all-MiniLM-L6-v2* encode sentences into compact semantic vectors. [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#:~:text=This%20is%20a%20sentence,like%20clustering%20or%20semantic%20search) [More on MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#:~:text=Our%20model%20is%20intended%20to,clustering%20or%20sentence%20similarity%20tasks) By building a system that retrieves relevant texts and images and feeds them to a language model (e.g., OPT-1.3B), you can answer queries that require understanding of both modalities.

### Objectives

In this lab you will:

* **Assemble a small multi-modal dataset** of images and associated text snippets or descriptions.
* **Compute embeddings** for images using CLIP and for text using a sentence transformer. Store embeddings in a FAISS index.
  [How CLIP and FAISS work](https://medium.com/@heyitssandeep/how-i-built-an-image-search-engine-with-clip-and-faiss-5b48df1df0fa#:~:text=What%20exactly%20is%20CLIP%20and,FAISS)
  [MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#:~:text=This%20is%20a%20sentence,like%20clustering%20or%20semantic%20search)
* **Implement a retrieval pipeline** that accepts a query (text or image), retrieves the most relevant cross-modal items from the index and feeds them into a language model.
* **Integrate a language model** (e.g., OPT-1.3B) to generate a final response based on the retrieved context.
  [OPT-1.3B](https://huggingface.co/facebook/opt-1.3b#:~:text=,models%20are%20available%20for%20study)
* **Handle cross-modal queries**, demonstrating that your system can answer questions like "Show me images related to this description" or "Describe the image content of this query".

### Task Overview

Your goal is to build an end-to-end RAG prototype that supports both text-to-image and image-to-text retrieval. You will:

1. **Collect a dataset** of 10–20 images and their associated texts (captions, descriptions or related paragraphs). Ensure each image has at least one corresponding text and that the mapping is clear.
2. **Generate embeddings** for images using CLIP and for texts using a sentence transformer (e.g., all-MiniLM-L6-v2). Normalize embeddings and store them in a FAISS index.
   [How CLIP and FAISS work](https://medium.com/@heyitssandeep/how-i-built-an-image-search-engine-with-clip-and-faiss-5b48df1df0fa#:~:text=What%20exactly%20is%20CLIP%20and,FAISS)
   [MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#:~:text=This%20is%20a%20sentence,like%20clustering%20or%20semantic%20search)
3. **Implement retrieval functions** that, given a query (text or image), compute its embedding and return the top-k similar items across modalities.
4. **Build a generation component** that takes retrieved items and uses a language model (such as OPT-1.3B) to craft a final response. For example, if the query is text-to-image, the response might describe the retrieved images; if image-to-text, it might summarize related text passages.
   [Multimodal RAG](https://www.meilisearch.com/blog/multimodal-rag#:~:text=Multimodal%20RAG%20shatters%20this%20limitation,world%20as%20humans%20experience%20it)
5. **Demonstrate cross-modal queries** and evaluate the relevance of retrieved items and the quality of generated responses.

## tip about library versions in the workspace

These versions of numpy and faiss-cpu have been known to work well together within the aws workspace:

```
numpy==1.24.3
faiss-cpu==1.7.4
torch
transformers
pillow
jupyter
```

### Requirements

Your submission must include:

* **A multi-modal dataset** of 10–20 images and associated text. Images and text should have clear relationships and be stored in a structured way (e.g., a folder of images and a CSV/JSON file with image file names and text).
* **FAISS index** containing embeddings for both modalities. Document the embedding models used and how you built the index.
* **Retrieval functions** for text-to-image, image-to-text and text-to-text queries.
* **Integration with a language model** (e.g., OPT-1.3B or another accessible LLM) to generate responses based on retrieved content.
* **Demonstration code** showing at least three example queries across different modalities and the system's outputs.
* **Brief report** describing the dataset, implementation details, evaluation of retrieval results, and reflections on challenges.

### Data Guidance

You will need to assemble your own small multi-modal dataset. Consider the following guidelines:

* **Size and structure**: Aim for 10–20 images. Each image must have one or more associated text descriptions or relevant paragraphs. Create a mapping file (e.g., CSV with columns `image_path`, `text`) to link them.
* **Sources**: Use openly licensed image repositories such as **Wikimedia Commons** or **Unsplash** for images and short articles or captions from **Wikipedia** or public dataset repositories for text. Ensure you respect copyright and licensing terms.
* **Relevance and variety**: Select images and texts that are diverse enough to demonstrate meaningful retrieval, such as photos of landmarks with encyclopedia descriptions, or product images with specifications.
* **Manageability**: Keep the dataset small enough to embed and search quickly but rich enough to show interesting relationships. Document any preprocessing steps (e.g., resizing images or cleaning text).

### Deliverables

Submit the following materials:

* **Code repository** (or notebook) containing:

  * Code for building the dataset (downloading or assembling images and text).
  * Code to compute CLIP and sentence-transformer embeddings and build the FAISS index.
  * Retrieval functions for cross-modal search.
  * Integration with a language model to generate answers.
  * Example queries and outputs.
* **Dataset files**: images and text mapping (e.g., a `data` folder with images and a `metadata.csv` or `metadata.json`).
* **Report** (1–2 pages) summarizing dataset creation, describing the system architecture, presenting example results, and discussing limitations and potential improvements.

### Steps & Recommendations

1. **Collect data.** Select 10–20 images and associated texts. Create a mapping file linking them. Use high-level categories (e.g., landmarks, animals, products) to make retrieval intuitive.
2. **Install models and dependencies.** Use Hugging Face's `openai/clip-vit-base-patch32` or similar to compute image embeddings and the `sentence-transformers/all-MiniLM-L6-v2` model for text embeddings. Install FAISS to store and search embeddings.
   [How CLIP and FAISS work](https://medium.com/@heyitssandeep/how-i-built-an-image-search-engine-with-clip-and-faiss-5b48df1df0fa#:~:text=What%20exactly%20is%20CLIP%20and,FAISS)
   [MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#:~:text=This%20is%20a%20sentence,like%20clustering%20or%20semantic%20search)

   We strongly recommend you use a virtual environment so you don't impact your global/base python install. Here are some recommended version numbers for libraries:
     numpy==1.24.3
     faiss-cpu==1.7.4
     torch
     transformers
     pillow
     jupyter 
4. **Compute embeddings.** Preprocess images (resize, normalize) and pass them through CLIP to obtain image vectors. Pass your texts through the sentence transformer to obtain text vectors. Save vectors with identifiers.
5. **Build FAISS index.** Create a single index or separate indices for images and texts. Use FAISS functions (e.g., `IndexFlatIP`) to add embeddings and perform similarity search. Normalize vectors so cosine similarity is equivalent to inner product.
6. **Implement retrieval functions.** Given a query (text or image), compute its embedding using the appropriate model and search the FAISS index. Return the top-k results across modalities. Consider merging results if you maintain separate indices.
7. **Integrate a language model.** For each query, compile the retrieved texts and image descriptions into a prompt. Use an LLM such as OPT-1.3B to generate a final answer. For example, for an image query, ask the model to describe the image using retrieved text passages; for a text query, ask the model to summarize the retrieved images and texts.
   [Multimodal RAG](https://www.meilisearch.com/blog/multimodal-rag#:~:text=Multimodal%20RAG%20shatters%20this%20limitation,world%20as%20humans%20experience%20it)
8. **Test cross-modal queries.** Run queries such as "Find an image of \[description]" or supply an image and ask "What is this and what related information exists?" Inspect the relevance of retrievals and the coherence of the generated answer.
9. **Reflect and document.** Evaluate how well your system handles cross-modal retrieval, where it succeeds and where it fails. Discuss potential improvements such as using more sophisticated fusion techniques or larger models.

### Extension / Stretch Goals

* **Any-to-any retrieval:** Extend your system to handle audio or other modalities by encoding additional types of data into the same vector space.
  [Multimodal RAG](https://www.meilisearch.com/blog/multimodal-rag#:~:text=Multimodal%20RAG%20shatters%20this%20limitation,world%20as%20humans%20experience%20it)
* **Advanced fusion strategies:** Explore weighting schemes or re-ranking methods to combine text and image retrieval results more effectively before generation.
* **Alternative embedding models:** Experiment with other sentence transformers or vision--language models (e.g., CLIP variants, BERT-like encoders) and compare retrieval quality.
* **Scalability:** Deploy your FAISS index as a service and experiment with larger datasets. Investigate approximate nearest neighbour (ANN) indices for scalability.

### References & Resources

* **CLIP and FAISS:** CLIP maps images and text to the same latent space, while FAISS provides efficient similarity search for dense embeddings.
  [CLIP & FAISS](https://medium.com/@heyitssandeep/how-i-built-an-image-search-engine-with-clip-and-faiss-5b48df1df0fa#:~:text=What%20exactly%20is%20CLIP%20and,FAISS)
* **Sentence transformers:** The all-MiniLM-L6-v2 model encodes sentences into 384-dimensional vectors suitable for semantic search.
  [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#:~:text=This%20is%20a%20sentence,like%20clustering%20or%20semantic%20search)
  [More on MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#:~:text=Our%20model%20is%20intended%20to,clustering%20or%20sentence%20similarity%20tasks)
* **Multimodal RAG:** Guides on multimodal RAG describe embedding, storing, retrieval, fusion and generation stages, highlighting cross-modal queries and any-to-any information retrieval.
  [Multimodal RAG](https://www.meilisearch.com/blog/multimodal-rag#:~:text=Multimodal%20RAG%20shatters%20this%20limitation,world%20as%20humans%20experience%20it)
* **OPT family**: Open pre-trained transformers like OPT-1.3B are decoder-only models available for research and can serve as the generative component in your RAG system.
  [OPT-1.3B](https://huggingface.co/facebook/opt-1.3b#:~:text=,models%20are%20available%20for%20study)
