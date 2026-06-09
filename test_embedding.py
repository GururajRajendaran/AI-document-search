from utils.embeddings import get_embedding

embedding = get_embedding(
    "Employees receive 12 casual leaves annually"
)

print(type(embedding))
print(len(embedding))
print(embedding[:5])