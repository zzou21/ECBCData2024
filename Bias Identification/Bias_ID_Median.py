import json
import os
import statistics
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Download the stopwords corpus if you haven't already
nltk.download('stopwords')

# Get the English stop words
stop_words = set(stopwords.words('english'))

# Make sure to download the punkt tokenizer
nltk.download('punkt')

# Load categories from JSON file
def load_categories(json_file):
    with open(json_file, 'r') as f:
        categories = json.load(f)
    return categories

# Function to read a document and break it into sentences
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
        
    text.lower()
    
    # Tokenize the text into sentences using nltk
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
    word_embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        for i, word in enumerate(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])):
            if word not in stop_words:
                word_embeddings.append(embeddings[0, i, :].numpy())
    return word_embeddings

def get_sentence_embedding(sentence, tokenizer, model):
    word_embeddings = {}
    chunk = sentence
    inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    for i, word in enumerate(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])):
        if word not in word_embeddings:
            word_embeddings[word] = embeddings[0, i, :].numpy()
    return word_embeddings

# Compute average vector for each category
def compute_category_embeddings(categories, tokenizer, model):
    categories_embeddings = {}
    for category, sentences in categories.items():
        category_embeddings = []
        for word, sentence in sentences.items():
            sentence_embedding = get_sentence_embedding(sentence, tokenizer, model)
            if word in sentence_embedding:
                term_embedding = sentence_embedding[word]
                category_embeddings.append(term_embedding)
        if category_embeddings:
            categories_embeddings[category] = np.mean(category_embeddings, axis=0)
        else:
            categories_embeddings[category] = np.zeros(model.config.hidden_size)
    return categories_embeddings


# Construct bias axes
def construct_bias_axes(category_embeddings):
    faith_bias_axis = category_embeddings["Faith"] - category_embeddings["Money"]
    return faith_bias_axis

# Function to project a word onto bias axes
def project_onto_bias_axis(embedding, bias_axis):
    return np.dot(embedding, bias_axis.T) / np.linalg.norm(bias_axis)

# Main function
def main(categories_json, document_path, model_name):
    # Load categories
    categories = load_categories(categories_json)
    
    # Load the MacBERTh model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Read and tokenize the document
    sentences = read_sentence_document(document_path)
    embeddings = get_word_embedding(sentences, tokenizer, model)

    # Compute category embeddings
    category_embeddings = compute_category_embeddings(categories, tokenizer, model)

    # Construct bias axes
    faith_bias_axis = construct_bias_axes(category_embeddings)
    
    # Example: Project words from the document onto bias axes
    projection_faith = [project_onto_bias_axis(embedding, faith_bias_axis) for embedding in embeddings]

    print(projection_faith)

    if (projection_faith is not None):
        print(f"{os.path.basename(document_path)}: ({statistics.median(projection_faith)})\n")
    

# Now this is the main; feel free to change the following directory where fit
base_dir = os.getcwd()

categories_json = os.path.join(base_dir, 'data/categorized_words.json')

model_name = 'emanjavacas/MacBERTh'

document_directory = os.path.join(base_dir, 'data/A00151.txt')

main(categories_json, document_directory, model_name)