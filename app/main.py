from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from app.embedding import embed_text
from app.cache import SemanticCache
from app.utils import load_dataset


# Load dataset
documents = load_dataset()
print("Loading dataset completed")

# Create embeddings for all documents
print("Embedding dataset...")
doc_vectors = embed_text(documents)
print("Embedding completed")


# Initialize FastAPI
app = FastAPI()

# Initialize cache
cache = SemanticCache()


# Request schema
class Query(BaseModel):
    query: str


# Semantic search function
def semantic_search(query_vector):

    similarities = cosine_similarity([query_vector], doc_vectors)[0]

    best_index = similarities.argmax()

    best_score = similarities[best_index]

    best_document = documents[best_index]

    return best_document, float(best_score)


# Query endpoint
@app.post("/query")
def query_search(q: Query):

    vector = embed_text([q.query])[0]

    hit, entry, sim = cache.search(vector)

    if hit:
        return {
            "query": q.query,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(sim),
            "result": entry["result"],
            "dominant_cluster": entry["cluster"]
        }

    # If cache miss → perform semantic search
    result, score = semantic_search(vector)

    cluster = 0

    cache.add(q.query, vector, result, cluster)

    return {
        "query": q.query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": score,
        "result": result,
        "dominant_cluster": cluster
    }


# Cache statistics
@app.get("/cache/stats")
def stats():
    return cache.stats()


# Clear cache
@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {"message": "Cache cleared"}
