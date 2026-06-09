import numpy as np

def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

def search_chunks(
    question_embedding,
    chunk_embeddings,
    chunks,
    top_k=3
):

    scores = []

    for i, chunk_embedding in enumerate(chunk_embeddings):

        similarity = cosine_similarity(
            question_embedding,
            chunk_embedding
        )

        scores.append(
            (similarity, chunks[i])
        )

    scores.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    return scores[:top_k]