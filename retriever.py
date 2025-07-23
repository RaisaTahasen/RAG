def retrieve_relevant_chunks(query, embed_model, vector_db):
    query_vec = embed_model([query])
    return vector_db.search(query_vec, top_k=5)


# def retrieve_relevant_chunks(query, embed_model, vector_db, top_k=5):
#     # Get semantic matches with scores
#     semantic_results = vector_db.search(embed_model([query]), top_k=top_k)
    
#     # Sort by combined score (70% semantic, 30% keyword overlap)
#     def keyword_score(text):
#         query_terms = query.split()
#         return sum(term in text for term in query_terms[:3]) / 3
    
#     return sorted(
#         semantic_results,
#         key=lambda x: 0.7 * x["similarity"] + 0.3 * keyword_score(x["text"]),
#         reverse=True
#     )[:top_k]
