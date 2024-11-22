import boto3
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Charger le modèle d'embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

def download_file_from_s3(bucket_name, file_name, local_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, file_name, local_path)

def create_embedding_from_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
        return model.encode(data)

def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array([embeddings]))
    return index
