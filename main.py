from preprocessor import extract_text_from_pdf, clean_text, chunk_text
from embedder import get_embeddings
from vector_store import VectorDB
from retriever import retrieve_relevant_chunks
from generator import generate_answer
from langchain_core.embeddings import Embeddings
import numpy as np

class CustomEmbeddings(Embeddings):
    def embed_documents(self, texts):
        # Convert to list if needed
        if isinstance(texts, str):
            texts = [texts]
        embeddings = get_embeddings(texts)
        return embeddings.cpu().numpy()  # Convert to list of lists
    
    def embed_query(self, text):
        return self.embed_documents(text)[0]  # Return first embedding

def main():
    # Load and chunk PDF
    text = extract_text_from_pdf("data/HSC26-Bangla1st-Paper.pdf")
    cleaned = clean_text(text)
    chunks = chunk_text(cleaned)

    # Create Vector DB
    embeddings = CustomEmbeddings()
    vector_db = VectorDB(embeddings)
    vector_db.add(chunks)

    # Chat loop
    chat_history = []
    while True:
        query = input("Ask something (Bangla or English): ")
        if query.lower() == 'exit':
            break
        retrieved = retrieve_relevant_chunks(query, embeddings, vector_db)
        answer = generate_answer(query, retrieved, chat_history)
        print("Answer:", answer)
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()