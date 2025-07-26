def retrieve_relevant_chunks(query, embed_model, vector_db, top_k=3):
    query_vec = embed_model.embed_query(query)
    return vector_db.search(query_vec, top_k=top_k)


# from sentence_transformers import CrossEncoder

# # Use a public model that doesn't require authentication
# RERANKER_MODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# def retrieve_relevant_chunks(query, embed_model, vector_db, top_k=3):
#     # First-stage retrieval (semantic search)
#     query_vec = embed_model.embed_query(query)
#     chunks = vector_db.search(query_vec, top_k=top_k*3)  # Get more candidates initially
    
#     # Rerank with cross-encoder
#     if len(chunks) > 1:
#         # Create query-chunk pairs for reranking
#         pairs = [(query, chunk) for chunk in chunks]
        
#         # Get reranker scores
#         reranker_scores = RERANKER_MODEL.predict(pairs)
        
#         # Combine with initial scores
#         ranked_chunks = sorted(
#             zip(chunks, reranker_scores),
#             key=lambda x: x[1],
#             reverse=True
#         )
        
#         # Return top chunks after reranking
#         return [chunk for chunk, score in ranked_chunks[:top_k]]
    
#     return chunks[:top_k]
#=================================



