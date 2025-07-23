import faiss
import numpy as np

class VectorDB:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, texts):
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query_vector, top_k=5):
        distances, indices = self.index.search(query_vector, top_k)
        return [self.texts[i] for i in indices[0]]
        # return [
        #     {
        #         "text": self.texts[i],
        #         "similarity": 1 - distances[0][j]  # Convert distance to similarity score
        #     }
        #     for j, i in enumerate(indices[0])
        # ]
