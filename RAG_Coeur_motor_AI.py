from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math
import numpy as np
import faiss


def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> list[str]:
    words = text.split()

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))

        if i + chunk_size >= len(words):
            break

    return chunks

def search(query: str, top_k: int = 3):
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        meta = chunk_metadata[idx]
        results.append({
            "score": float(score),
            "doc_id": meta["doc_id"],
            "chunk_id": meta["chunk_id"],
            "text": meta["text"]
        })
    return results


if __name__ == "__main__":
    
    documents = [
    """
    Python is widely used in artificial intelligence and machine learning.
    It is popular because it has many libraries like TensorFlow, PyTorch,
    and scikit-learn. Python is also easy to learn for beginners.
    """,
    """
    PostgreSQL is a relational database system. It stores structured data
    and is used in many web applications. SQL is the language used to query
    and manage relational databases.
    """,
    """
    Linux is an open source operating system. It is widely used on servers,
    by developers, and in cybersecurity environments. Many programmers like Linux
    because it is flexible and powerful.
    """
]

    all_chunks = []
    chunk_metadata = []

    for doc_id, doc in enumerate(documents):
        chunks = chunk_text(doc, chunk_size=30, overlap=8)
        for chunk_id, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": chunk
            })

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 4) Encoder les chunks
    embeddings = model.encode(all_chunks, convert_to_numpy=True)

    # 5) Convertir en float32 pour FAISS
    embeddings = embeddings.astype("float32")

    # 6) Normaliser les vecteurs
    faiss.normalize_L2(embeddings)

    # 7) Créer l'index FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
#   creation d une matrice de vecteur avec n dimesion  
#   C’est l’index le plus simple :
#   Flat : recherche exacte
#   IP : inner product
#   Donc on compare la question avec tous les vecteurs.

    # 8) Ajouter les vecteurs à l'index
    index.add(embeddings)

    print(f"Nombre total de chunks indexés : {index.ntotal}")
    print(f"Dimension des vecteurs : {dimension}")

    query = "Which language is used in artificial intelligence?"
    results = search(query, top_k=3)

    print("\nQuestion:", query)
    print()

    for r in results:
        print(f"Score: {r['score']:.4f}")
        print(f"Document: {r['doc_id']} | Chunk: {r['chunk_id']}")
        print(r["text"])
        print("-" * 60)