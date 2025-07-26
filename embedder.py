from sentence_transformers import SentenceTransformer

#model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')  # Better for Bangla
#model = SentenceTransformer('sagorsarker/bangla-bert-base')
def get_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)
