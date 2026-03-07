from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:

    def __init__(self, threshold=0.85):
        self.cache = []
        self.hit = 0
        self.miss = 0
        self.threshold = threshold

    def search(self, query_vector):

        for entry in self.cache:
            similarity = cosine_similarity([query_vector],[entry["vector"]])[0][0]

            if similarity > self.threshold:
                self.hit += 1
                return True, entry, similarity

        self.miss += 1
        return False, None, 0

    def add(self, query, vector, result, cluster):

        self.cache.append({
            "query": query,
            "vector": vector,
            "result": result,
            "cluster": cluster
        })

    def stats(self):

        total = self.hit + self.miss

        return {
            "total_entries": len(self.cache),
            "hit_count": self.hit,
            "miss_count": self.miss,
            "hit_rate": self.hit/total if total else 0
        }

    def clear(self):
        self.cache = []
        self.hit = 0
        self.miss = 0
