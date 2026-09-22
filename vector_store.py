from embeddings import encode_image_clip, encode_text_clip, encode_text_sentence_transformer

import faiss
import numpy as np

faiss.omp_set_num_threads(1)


class MultiModalIndex:
    def __init__(self, records):
        self.records = records
        self.image_clip_index = None
        self.text_clip_index = None
        self.text_st_index = None

    def build(self):
        image_vectors = np.stack(
            [encode_image_clip(record["image_path"]) for record in self.records]
        )
        text_clip_vectors = np.stack(
            [encode_text_clip(record["text"]) for record in self.records]
        )
        text_st_vectors = np.stack(
            [encode_text_sentence_transformer(record["text"]) for record in self.records]
        )

        self.image_clip_index = faiss.IndexFlatIP(image_vectors.shape[1])
        self.image_clip_index.add(image_vectors)

        self.text_clip_index = faiss.IndexFlatIP(text_clip_vectors.shape[1])
        self.text_clip_index.add(text_clip_vectors)

        self.text_st_index = faiss.IndexFlatIP(text_st_vectors.shape[1])
        self.text_st_index.add(text_st_vectors)

    def _search(self, index, query_vector, k):
        scores, indices = index.search(np.expand_dims(query_vector, 0), k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.records[idx], float(score)))
        return results

    def retrieve_images_for_text(self, query_text, k=3):
        query_vector = encode_text_clip(query_text)
        return self._search(self.image_clip_index, query_vector, k)

    def retrieve_texts_for_image(self, image_path, k=3):
        query_vector = encode_image_clip(image_path)
        return self._search(self.text_clip_index, query_vector, k)

    def retrieve_texts_for_text(self, query_text, k=3):
        query_vector = encode_text_sentence_transformer(query_text)
        return self._search(self.text_st_index, query_vector, k)
