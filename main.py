from preprocessor import extract_text_from_pdf, clean_text, chunk_text
from embedder import get_embeddings
from vector_store import VectorDB
from retriever import retrieve_relevant_chunks
from generator import generate_answer


def main():
    # Load and chunk PDF

    text = extract_text_from_pdf("data/HSC26-Bangla1st-Paper.pdf")
    cleaned = clean_text(text)
    chunks = chunk_text(cleaned)

    # Embed and store in Vector DB
    embeddings = get_embeddings(chunks)
    vector_db = VectorDB(embeddings[0].shape[0])
    vector_db.add(embeddings.cpu().numpy(), chunks)

    # Chat loop
    chat_history = []
    while True:
        query = input("Ask something (Bangla or English): ")
        if query.lower() == 'exit':
            break
        retrieved = retrieve_relevant_chunks(query, get_embeddings, vector_db)
        answer = generate_answer(query, retrieved, chat_history)
        print("Answer:", answer)
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
