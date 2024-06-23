import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import nltk

# Ensure you have the NLTK punkt tokenizer downloaded
nltk.download('punkt')

# Load categories from JSON file
def load_categories(json_file):
    with open(json_file, 'r') as f:
        categories = json.load(f)
    return categories

# Function to read a document and break it into sentences
def read_and_tokenize_document(document_path):
    with open(document_path, 'r') as f:
        text = f.read()
    sentences = nltk.sent_tokenize(text)
    return sentences

# Function to get word embeddings
def get_word_embedding(word, tokenizer, model):
    inputs = tokenizer(word, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

# Compute average vector for each category
def compute_category_embeddings(categories, tokenizer, model):
    category_embeddings = {}
    for category, words in categories.items():
        embeddings = []
        for word in words:
            embedding = get_word_embedding(word, tokenizer, model)
            embeddings.append(embedding)
        category_embeddings[category] = np.mean(embeddings, axis=0)
    return category_embeddings

# Construct bias axes
def construct_bias_axes(category_embeddings):
    faith_bias_axis = category_embeddings["Faith"] - category_embeddings["Money"]
    like_bias_axis = category_embeddings["Attraction"] - category_embeddings["Repulsion"]
    return faith_bias_axis, like_bias_axis

# Function to project a word onto bias axes
def project_onto_bias_axes(word, tokenizer, model, faith_bias_axis, money_bias_axis):
    embedding = get_word_embedding(word, tokenizer, model)
    projection_faith = np.dot(embedding, faith_bias_axis.T) / np.linalg.norm(faith_bias_axis)
    projection_money = np.dot(embedding, money_bias_axis.T) / np.linalg.norm(money_bias_axis)
    return projection_faith, projection_money

# Main function
def main(categories_json, document_path, model_name):
    # Load categories
    categories = load_categories(categories_json)
    
    # Load the MacBERTh model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Read and tokenize the document
    sentences = read_and_tokenize_document(document_path)
    
    # Compute category embeddings
    category_embeddings = compute_category_embeddings(categories, tokenizer, model)
    
    # Construct bias axes
    faith_bias_axis, money_bias_axis = construct_bias_axes(category_embeddings)
    
    # Example: Project words from the document onto bias axes
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            projection_faith, projection_money = project_onto_bias_axes(word, tokenizer, model, faith_bias_axis, money_bias_axis)
            print(f"Projection of '{word}' onto Faith-Repulsion axis: {projection_faith}")
            print(f"Projection of '{word}' onto Money-Attraction axis: {projection_money}")

# Now this is the main; feel free to change the following directory where fit
categories_json = 'data/categorized_words.json'
document_path = 'path/to/your/document.txt'
model_name = "data/fine-tuned-MacBERTh"
# model_name = "emanjavacas/MacBERTh"
main(categories_json, document_path, model_name)
