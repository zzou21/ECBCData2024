from transformers import AutoTokenizer, AutoModel
import string, numpy as np
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
                word_embeddings[word] = embeddings[0, i, :].cpu().numpy()
            else:
                word_embeddings[word] = (word_embeddings[word] * (word_times[word]-1) + embeddings[0, i, :].cpu().numpy()) / word_times[word]
    
    return word_embeddings

def clean_embedding(embeddings, tokenizer, stop_words):
    clean = {}
    for token, embedding in embeddings.items():
        if (token in tokenizer.all_special_tokens) or (token.lower() in stop_words) or (token in string.punctuation) or (token.startswith('##')) or (token == "•"):
            continue
        clean[token] = embedding
    return clean

# Main function
def process_file(file_name, folderPath, tokenizer, model):
    document_path = os.path.join(folderPath, file_name)
    
    if os.path.isfile(document_path) and not file_name.startswith("."):
        sentences = read_sentence_document(document_path)
        docEmbedding = clean_embedding(get_word_embedding(sentences, tokenizer, model), tokenizer, stop_words)
        return file_name, docEmbedding
    return None

model_name = "emanjavacas/MacBERTh"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model = torch.nn.DataParallel(model)  # Enable multi-GPU support
model = model.cuda()  # Move model to GPU
stop_words = set(stopwords.words('english'))

base_dir = os.path.dirname(os.path.abspath(__file__))
folderPath = os.path.join(base_dir, "../AllVirginia")

# List all files in the folder
file_names = [f for f in os.listdir(folderPath) if os.path.isfile(os.path.join(folderPath, f)) and not f.startswith(".")]

# Parallel processing of files
results = Parallel(n_jobs=4)(delayed(process_file)(file_name, folderPath, tokenizer, model) for file_name in file_names)

# Filter out None results
results = [result for result in results if result is not None]

# Combine results into a single dictionary
all_embeddings = {file_name: embedding for file_name, embedding in results}

# Save all embeddings to a JSON file
output_path = os.path.join(base_dir, "VA_embeddings.json")
with open(output_path, 'w') as f:
    json.dump(all_embeddings, f)

print(f"Processed {len(all_embeddings)} files.")
