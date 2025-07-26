from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from typing import List

class VectorDB:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.db = Chroma(
            embedding_function=embeddings,
            collection_name="bangla_rag",
            persist_directory="./chroma_db"  # Optional persistence
        )
        self.texts = []

    def add(self, texts: List[str]):
        """Add texts to ChromaDB with automatic embedding"""
        self.db.add_texts(texts)
        self.texts.extend(texts)  # Keep reference for reranking

    def search(self, query_vector, top_k=5) -> List[str]:
        """Search using Chroma's native similarity search"""
        # Convert vector to list if it's numpy array
        if hasattr(query_vector, 'tolist'):
            query_vector = query_vector.tolist()
            
        results = self.db.similarity_search_by_vector(
            embedding=query_vector,
            k=top_k*3  # Get more candidates for reranking
        )
        return [doc.page_content for doc in results]


# import faiss
# import numpy as np

# class VectorDB:
#     def __init__(self, embeddings):
#         self.embeddings = embeddings
#         # Get dimension from a test embedding
#         test_embed = self.embeddings.embed_query("test")
#         self.dim = len(test_embed)
#         self.index = faiss.IndexFlatL2(self.dim)  # Initialize index here
#         self.texts = []

#     def add(self, texts):
#         # Get embeddings for the texts
#         embeddings = self.embeddings.embed_documents(texts)
#         # Convert to numpy array if needed
#         if not isinstance(embeddings, np.ndarray):
#             embeddings = np.array(embeddings)
#         # Ensure correct shape (n_samples, n_features)
#         if len(embeddings.shape) == 1:
#             embeddings = embeddings.reshape(1, -1)
#         # Add to index
#         self.index.add(embeddings)
#         self.texts.extend(texts)

#     def search(self, query_vector, top_k=5):
#         # Convert to numpy array if needed
#         if not isinstance(query_vector, np.ndarray):
#             query_vector = np.array(query_vector)
#         # Ensure correct shape (1, n_features)
#         if len(query_vector.shape) == 1:
#             query_vector = query_vector.reshape(1, -1)
#         # Search the index
#         distances, indices = self.index.search(query_vector, top_k)
#         return [self.texts[i] for i in indices[0]]