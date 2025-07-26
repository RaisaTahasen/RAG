# api.py
from flask import Flask, request, jsonify
from flask_cors import CORS 
from preprocessor import extract_text_from_pdf, clean_text, chunk_text
from embedder import get_embeddings
from vector_store import VectorDB
from retriever import retrieve_relevant_chunks
from generator import generate_answer
from langchain_core.embeddings import Embeddings
import numpy as np

app = Flask(__name__)
CORS(app) 
class CustomEmbeddings(Embeddings):
    def embed_documents(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = get_embeddings(texts)
        return embeddings.cpu().numpy()
    
    def embed_query(self, text):
        return self.embed_documents(text)[0]

# Initialize the RAG system at startup
embeddings = CustomEmbeddings()
vector_db = VectorDB(embeddings)

# Load and index the PDF (could be moved to a separate initialization endpoint)
text = extract_text_from_pdf("data/HSC26-Bangla1st-Paper.pdf")
cleaned = clean_text(text)
chunks = chunk_text(cleaned)
vector_db.add(chunks)

# Store conversation history per session (in-memory, consider Redis for production)
conversation_histories = {}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    
    # Validate input
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request"}), 400
    
    query = data['query']
    session_id = data.get('session_id', 'default')
    
    # Get or initialize chat history for this session
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    chat_history = conversation_histories[session_id]
    
    # Process the query
    retrieved = retrieve_relevant_chunks(query, embeddings, vector_db)
    answer = generate_answer(query, retrieved, chat_history)
    
    # Update chat history
    conversation_histories[session_id].extend([
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer}
    ])
    
    # Prepare response
    response = {
        "query": query,
        "answer": answer,
        "session_id": session_id
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)