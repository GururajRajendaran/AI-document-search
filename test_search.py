
from utils.embeddings import get_embedding
from utils.search import search_chunks

chunks = [
    "Employees receive 12 casual leaves annually.",
    "Work from home is allowed three days per week.",
    "Office parking is available for employees."
]

chunk_embeddings = [
    get_embedding(chunk)
    for chunk in chunks
]

question = "How many leave days do employees get?"

question_embedding = get_embedding(
    question
)

results = search_chunks(
    question_embedding,
    chunk_embeddings,
    chunks
)

for score, chunk in results:

    print(
        f"Score: {score:.4f}"
    )

    print(chunk)

    print("-" * 50)