import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from concurrent.futures import ProcessPoolExecutor, as_completed

# Make sure to download the punkt tokenizer
nltk.download('punkt')

# Load categories from JSON file
def load_categories(json_file):
    with open(json_file, 'r') as f:
        categories = json.load(f)
    return categories

# Function to read a document and break it into sentences
def read_sentence_document(document_path):
    with open(document_path, 'r') as f:
        text = f.read()
        text = text.lower()
    
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
def get_word_embedding(chunk, tokenizer, model):
    word_embeddings = {}
    inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    for i, word in enumerate(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])):
        if word not in word_embeddings:
            word_embeddings[word] = embeddings[0, i, :].numpy()
    return word_embeddings

def get_single_embedding(word, tokenizer, model):
    # Tokenize the word
    inputs = tokenizer(word, return_tensors='pt')
    
    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the embedding for the word
    embeddings = outputs.last_hidden_state
    word_embedding = embeddings[0, 1, :].numpy()  # [CLS] token is at index 0, the word is at index 1
    return word_embedding

# Compute average vector for each category
def compute_category_embeddings(categories, tokenizer, model):
    categories_embeddings = {}
    for category, words in categories.items():
        category_embeddings = []
        for word in words:
            term_embedding = get_single_embedding(word, tokenizer, model)
            category_embeddings.append(term_embedding)
        if category_embeddings:
            categories_embeddings[category] = np.mean(category_embeddings, axis=0)
    return categories_embeddings

# Construct bias axes
def construct_bias_axes(category_embeddings):
    faith_bias_axis = category_embeddings["Faith"] - category_embeddings["Money"]
    desire_bias_axis = category_embeddings["Attraction"] - category_embeddings["Repulsion"]
    return faith_bias_axis, desire_bias_axis

# Function to project a word onto bias axes
def project_onto_bias_axis(word, embeddings, bias_axis, tokenizer, model):
    if word in embeddings:
        embedding = embeddings[word]
        projection = np.dot(embedding, bias_axis.T) / np.linalg.norm(bias_axis)
        return projection
    else:
        embedding = get_single_embedding(word, tokenizer, model)
        projection = np.dot(embedding, bias_axis.T) / np.linalg.norm(bias_axis)
        return projection

# Main function
def process_document(args):
    categories_json, document_path, model_name, keyword = args
    
    # Load categories
    categories = load_categories(categories_json)
    
    # Load the MacBERTh model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Read and tokenize the document
    sentences = read_sentence_document(document_path)
    embeddings = {}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(get_word_embedding, chunk, tokenizer, model) for chunk in sentences]
        for future in as_completed(futures):
            result = future.result()
            embeddings.update(result)
    
    # Compute category embeddings
    category_embeddings = compute_category_embeddings(categories, tokenizer, model)
    
    # Construct bias axes
    faith_bias_axis, desire_bias_axis = construct_bias_axes(category_embeddings)
    
    # Example: Project words from the document onto bias axes
    projection_faith = project_onto_bias_axis(keyword, embeddings, faith_bias_axis, tokenizer, model)
    projection_desire = project_onto_bias_axis(keyword, embeddings, desire_bias_axis, tokenizer, model)

    if (projection_faith is not None) and (projection_desire is not None):
        print(f"({projection_faith}, {projection_desire})")

# Now this is the main; feel free to change the following directory where fit
base_dir = os.path.dirname(os.path.abspath(__file__))

keyword = "profit"
categories_json = "categorized_words.json"
model_name = "finetuned_MacBERTh_Bible"
document_directory = os.path.join(base_dir, '..', 'EEBOphase2_1590-1639_body_texts')

args_list = [(categories_json, os.path.join(document_directory, file_name), model_name, keyword) 
             for file_name in os.listdir(document_directory) 
             if os.path.isfile(os.path.join(document_directory, file_name))]

with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_document, args) for args in args_list]
    for future in as_completed(futures):
        future.result()
