import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

_text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
_text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME)
_text_model.eval()

_clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
_clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
_clip_model.eval()


def _normalize(vector):
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.clip(norm, a_min=1e-10, a_max=None)


def encode_text_sentence_transformer(text):
    inputs = _text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = _text_model(**inputs).last_hidden_state.mean(dim=1)
    return _normalize(output.numpy())[0].astype("float32")


def encode_text_clip(text):
    inputs = _clip_processor(
        text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77
    )
    with torch.no_grad():
        output = _clip_model.get_text_features(**inputs).pooler_output
    return _normalize(output.numpy())[0].astype("float32")


def encode_image_clip(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = _clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = _clip_model.get_image_features(**inputs).pooler_output
    return _normalize(output.numpy())[0].astype("float32")
