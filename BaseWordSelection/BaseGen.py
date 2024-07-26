
from transformers import AutoTokenizer, AutoModel
import string, heapq, numpy as np
from torch.nn.functional import cosine_similarity
from nltk.corpus import stopwords
import os, json
import torch
import nltk
from nltk.tokenize import sent_tokenize
from joblib import Parallel, delayed

def read_sentence_document(document_path):
    encodings = ['utf-8', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(document_path, 'r', encoding=encoding) as f:
                text = f.read()
            break  # Exit the loop if no exception was raised
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("Failed to read the file with any of the tried encodings.")
        
    text = text.lower()

    sentences = sent_tokenize(text)

    # Further split the sentences if there are more than 450 words
    split_sentences = []
    for sentence in sentences:
        words = sentence.split()
        while len(words) > 400:
            split_sentences.append(' '.join(words[:400]))
            words = words[400:]
        split_sentences.append(' '.join(words))

    # Strip whitespace and filter out empty sentences
    final_sentences = [s.strip() for s in split_sentences if s.strip()]
    
    return final_sentences

# Function to get word embeddings
def get_word_embedding(chunks, tokenizer, model):
    word_times = {}
    word_embeddings = {}
    for chunk in chunks:

        inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        for i, word in enumerate(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])):
            if word not in word_times:
                word_times[word] = 1
            else:
                word_times[word] += 1
            if word not in word_embeddings:
                word_embeddings[word] = embeddings[0, i, :].numpy()
            else:
                word_embeddings[word] = (word_embeddings[word] * (word_times[word]-1) + embeddings[0, i, :].numpy()) / word_times[word]
    return word_embeddings

def get_single_embedding(word, tokenizer, model):
    # Tokenize the input word
    inputs = tokenizer(word, return_tensors='pt')
    print(f"Embedding word: {word}")
    
    # Disable gradient calculation
    with torch.no_grad():
        # Pass the tokenized input through the model
        outputs = model(**inputs)
    
    # Extract the embeddings
    embeddings = outputs.last_hidden_state
    
    # Convert token IDs back to tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Store the embeddings in a dictionary
    word_embeddings = {}
    for i, token in enumerate(tokens):
        if token == word:
            word_embeddings[token] = embeddings[0, i, :].numpy()
            break
    
    return word_embeddings

# Find the substitute words for the original keyword, by iterating over the standard word list
def substitute_word(word):
    json_dir = os.path.join(base_dir, "../standardizedwords.json")
    with open (json_dir, "r") as f:
        standardWord = json.load(f)
    ret = []
    for term, equals in standardWord.items():
        for spell in equals:
            if spell == word:
                ret.append(term)
    return ret

def clean_embedding(embeddings):
    clean = {}
    for token, embedding in embeddings.items():
        if (token in tokenizer.all_special_tokens) or (token.lower() in stop_words) or (token in string.punctuation) or (token.startswith('##')) or (token == "•"):
            continue
        clean[token] = embedding
    return clean

def process_key_word(keyWord, embeddings, tokenizer, model, top_n):
    presence = ""
    wordFound = False

    if keyWord in embeddings:
        presence = keyWord
        wordFound = True
    else:
        for equal in substitute_word(keyWord):
            wordFound = wordFound or equal in embeddings
            if wordFound:
                presence = equal
                break

    if not wordFound:
        key_embedding = get_single_embedding(keyWord, tokenizer, model)[keyWord]
    else:
        key_embedding = embeddings[presence]

    similarities = {}

    for token, embedding in embeddings.items():
        similarities[token] = np.dot(embedding, key_embedding) / ((np.linalg.norm(embedding)) * (np.linalg.norm(key_embedding)))

    # Use a heap to find the top N similar words
    top_entries = []

    for token, similarity in similarities.items():
        heapq.heappush(top_entries, (similarity, token))

        if len(top_entries) > top_n:
            removed_sim, removed_t = heapq.heappop(top_entries)

    top_entries.sort(reverse=True, key=lambda x: x[0])

    return keyWord, top_entries

# Main function
def find_top_similar_words(target_words, sentences, tokenizer, model, top_n):
    embeddings = clean_embedding(get_word_embedding(sentences, tokenizer, model))
    embeddingCoordinates = {word: coordiantes for word, coordiantes in embeddings.items()}

    # Use joblib to parallelize the processing of keywords
    results = Parallel(n_jobs=10)(delayed(process_key_word)(keyWord, embeddings, tokenizer, model, top_n) for keyWord in target_words)
    
    # Convert results to a dictionary
    key_sim_word = {key: value for key, value in results}
    
    return key_sim_word, embeddingCoordinates

model_name = "emanjavacas/MacBERTh"
# model_name = "fine-tuned-MacBERTh"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
stop_words = set(stopwords.words('english'))


base_dir = os.path.dirname(os.path.abspath(__file__))
# Path to the .txt file
# file_path = os.path.join(base_dir, "../VirginiaTotal.txt")
file_path = os.path.join(base_dir, "../data/VirginiaTotal.txt")
# file_path = "data/A10010_cleaned.txt"

sentences = read_sentence_document(file_path)

# Target word to find similarities with
target_words = ["native"]

# Find and print top 10 similar words
top_similar_words, embeddingCoordinates = find_top_similar_words(target_words, sentences, tokenizer, model, 50)

# embeddingCoordinatesStorage = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/BaseWordSelection/embeddingCoordinates.json" #path to JSON file that stores the embedding locations

# def save_embeddings_to_json(data, file_path):
#     # Convert numpy arrays to lists
#     def convert_ndarray(obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         raise TypeError("Object of type %s is not JSON serializable" % type(obj).__name__)
    
#     with open(file_path, 'w') as f:
#         json.dump(data, f, default=convert_ndarray)

# save_embeddings_to_json(embeddingCoordinates, embeddingCoordinatesStorage)

for word, words in top_similar_words.items():
    print("######################################################")
    print(f"Top 50 words most similar to '{word}':")
    for (sim, relWord) in words:
        print(f"{relWord}: {sim}")
