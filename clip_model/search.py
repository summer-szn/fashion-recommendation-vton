import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import torch
import numpy as np
import time
import faiss
from transformers import CLIPTokenizerFast, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "cuda not detected"

model = CLIPModel.from_pretrained("clip_model/clip-finetuned", torch_dtype=torch.float16).to(device)
processor = CLIPTokenizerFast.from_pretrained("clip_model/clip-finetuned", use_fast=True)

image_embeddings = np.load("clip_model/image_embeddings.npy").astype('float32')
with open("clip_model/image_filenames.json", "r") as f:
    image_filenames = json.load(f)

dimension = image_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension) 
index.add(image_embeddings) 

def execute_ultra_fast_search(query_text, top_k=10):
    inputs = processor([query_text], return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        _ = model.get_text_features(**inputs)
    
    start_time = time.perf_counter()    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features.cpu().numpy().astype('float32')
        faiss.normalize_L2(text_features)
    scores, indices = index.search(text_features, top_k)
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    
    print(f"\nLatency: {latency_ms:.4f} ms")
    for i, idx in enumerate(indices[0]):
        print(f"Rank {i+1}: {image_filenames[idx]} | Confidence Score: {scores[0][i]:.4f}")