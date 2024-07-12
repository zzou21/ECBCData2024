import json
import numpy as np
import heapq, string
from nltk.corpus import stopwords
from transformers import AutoTokenizer

# Load the JSON data
with open('data/embeddingCoordinates.json', 'r') as f:
    word_embeddings = json.load(f)

stop_words = set(stopwords.words('english'))

# Initialize the tokenizer
model_name = "emanjavacas/MacBERTh"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Function to find the top 50 similar words using a heap
def find_top_similar_words(input_word, embeddings, top_n=50):
    input_vector = np.array(embeddings[input_word])
    heap = []

    for word, vector in embeddings.items():
        # Filter out special tokens, stop words, punctuation, subwords, and specific non-standard characters
        if (word in tokenizer.all_special_tokens or
            word.lower() in stop_words or
            word in string.punctuation or
            word.startswith('##') or
            word in ["•", "☊"]):
            continue
        
        similarity = cosine_similarity(input_vector, np.array(vector))
        # Use a negative similarity because heapq is a min-heap
        heapq.heappush(heap, (similarity, word))
        if len(heap) > top_n:
            heapq.heappop(heap)
        
    # Extract and sort the words and their similarities
    top_similar_words = heap
    top_similar_words.sort(reverse=True, key=lambda x: x[0])  # Reverse to get the words in descending order of similarity

    return top_similar_words

# Example usage
input_word = "dispossess"

top_similar_words = find_top_similar_words(input_word, word_embeddings)
print(f"Top 50 words similar to '{input_word}': {top_similar_words}")
