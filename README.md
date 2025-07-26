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

For Web Interface:  

Flask: For creating the web API.  

Flask-CORS: To handle Cross-Origin Resource Sharing (CORS) for the web interface.  

python-dotenv: For managing environment variables (e.g., API keys).  
