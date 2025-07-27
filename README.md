This is a bilingual Retrieval-Augmented Generation (RAG) pipeline that supports queries in both **Bangla** and **English**. It retrieves relevant chunks from a Bangla document and generates context-aware answers using a language model.
## 🚀 Features

- 🔍 Semantic search with multilingual support (Bangla + English)
- 🧩 Bangla-aware chunking using LlamaIndex
- 🧠 Uses `paraphrase-multilingual-mpnet-base-v2` for better cross-lingual embeddings
- 🧑‍💻 Simple web UI to chat with your RAG model  

For detailed instructions about setup, please refer to the PDF file in the repository:
📄 [Setup_Guide.pdf](https://github.com/RaisaTahasen/RAG/blob/main/Setup%20Guide.pdf)  
Please ensure the "data" folder contains the pdf "HSC26 Bangla 1st paper".  

For the questions and answers required for the assessment, please refer to [RAG_Q/A.pdf](https://github.com/RaisaTahasen/RAG/blob/main/RAG_Q_A.pdf)  

📌 Credits

-LlamaIndex

-Sentence Transformers

-OpenRouter

-ChromaDB

**Used Tools, Libraries, and Packages**  
Core Libraries:  

For PDF Processing:  

PyMuPDF (pymupdf): For fast and efficient text extraction from PDFs.
LlamaIndex (SentenceSplitter): For advanced text chunking with support for Bengali punctuation.
Embeddings and Vector Storage:Sentence Transformers: For generating multilingual embeddings (e.g., paraphrase-multilingual-mpnet-base-v2).
LangChain Chroma: For vector storage and similarity search.  

NLP and RAG:  

LangChain: For orchestration, embeddings, and chat models.  
OpenAI (via OpenRouter): For generating answers using anthropic/claude-3-haiku:beta.  
HuggingFace Transformers: For embeddings and reranking (experimented not implemented in the current version)  

Utilities:  

NumPy: For numerical operations on embeddings.  
Torch: For tensor operations (used with Sentence Transformers).  

## For Web Interface:  

Flask: For creating the web API.  

Flask-CORS: To handle Cross-Origin Resource Sharing (CORS) for the web interface.  

python-dotenv: For managing environment variables (e.g., API keys).  


**📡 API Documentation**  
The RAG Chat API is a session-aware service designed to handle both English and Bangla user queries. It operates over HTTP and is currently configured for local development.  

Endpoint  
The API is accessible via a POST request to the /chat route on the local server,hosted at http://127.0.0.1:5000/chat.  

Request Parameters  
The POST request must include a JSON payload with two required fields:  

Query: This is the user’s message or question, written in either English or Bangla.  

Session ID: A unique identifier used to maintain the conversation context across multiple interactions.  

Clients are responsible for generating and storing this session ID to preserve continuity in the chat experience.  

Response
On a successful request, the API returns a JSON response containing an answer field. This field holds the AI-generated reply based on the query and previous context associated with the session.  

Error Handling  
The API returns standard HTTP status codes to indicate the result of a request:  

A 400 Bad Request error is returned when the input JSON is missing required fields or is improperly formatted.  

A 500 Internal Server Error indicates that something went wrong during the processing of the request.  

Development Notes
The session context is stored temporarily in memory, meaning it is lost if the server restarts.



