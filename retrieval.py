import numpy as np

def search_documents(query, index, model, top_k=5):
    query_embedding = model.encode(query)
    D, I = index.search(np.array([query_embedding]), k=top_k)
    return D, I
