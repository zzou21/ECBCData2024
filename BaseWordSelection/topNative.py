import json
import numpy as np
import heapq

with open("data/embeddingCoordinates.json", "r") as f:
    embeddings = json.load(f)

native = embeddings["native"]

similarities = {}

for token, embedding in embeddings.items():
    similarities[token] = np.dot(embedding, native) / ((np.linalg.norm(embedding)) * (np.linalg.norm(native)))

    # Use a heap to find the top N similar words
    top_entries = []

    for token, similarity in similarities.items():
        heapq.heappush(top_entries, (similarity, token))

        if len(top_entries) > 100:
            removed_sim, removed_t = heapq.heappop(top_entries)

    top_entries.sort(reverse=True, key=lambda x: x[0])


print(top_entries)
