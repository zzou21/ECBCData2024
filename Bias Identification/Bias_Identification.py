import json
import os, re
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
def read_sentence_document(document_path):
    with open(document_path, 'r') as f:
        text = f.read()
    pattern = r'(?<!\.)\.(?!\.)\s*|(?<=\!|\?)\s*|(?<=\.\.\.)\s*|(?<=\:)\s*|(?<=,)\s*'
    sentences = re.split(pattern, text)
    return [sentences.strip() for sentence in sentences if sentence.strip()]

# Function to get word embeddings
def get_word_embedding(chunks, tokenizer, model):
    word_embeddings = {}
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        for i, word in enumerate(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])):
            if word not in word_embeddings:
                word_embeddings[word] = embeddings[0, i, :].numpy()
    return word_embeddings

# Compute average vector for each category
def compute_category_embeddings(categories, embeddings, tokenizer, model):
    categories_embeddings = {}
    for category, words in categories.items():
        category_embeddings = []
        for word in words:
            embedding = embeddings[word]
            category_embeddings.append(embedding)
        categories_embeddings[category] = np.mean(category_embeddings, axis=0)
    return categories_embeddings

# Construct bias axes
def construct_bias_axes(category_embeddings):
    faith_bias_axis = category_embeddings["Faith"] - category_embeddings["Money"]
    desire_bias_axis = category_embeddings["Attraction"] - category_embeddings["Repulsion"]
    return faith_bias_axis, desire_bias_axis

# Function to project a word onto bias axes
def project_onto_bias_axis(word, embeddings, bias_axis):
    embedding = embeddings(word)
    projection = np.dot(embedding, bias_axis.T) / np.linalg.norm(bias_axis)
    return projection

# Main function
def main(categories_json, document_path, model_name, keyword):
    # Load categories
    categories = load_categories(categories_json)
    
    # Load the MacBERTh model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Read and tokenize the document
    sentences = read_sentence_document(document_path)
    embeddings = get_word_embedding(sentences, tokenizer, model)
    
    # Compute category embeddings
    category_embeddings = compute_category_embeddings(categories, embeddings, tokenizer, model)
    
    # Construct bias axes
    faith_bias_axis, desire_bias_axis = construct_bias_axes(category_embeddings)
    
    # Example: Project words from the document onto bias axes
    projection_faith = project_onto_bias_axis(keyword, embeddings, faith_bias_axis)
    projection_desire = project_onto_bias_axis(keyword, embeddings, desire_bias_axis)

    print(f"Projection of '{keyword}' onto Faith-Money axis: {projection_faith}")
    print(f"Projection of '{keyword}' onto Attraction-Repulsion axis: {projection_desire}")



# Now this is the main; feel free to change the following directory where fit
base_dir = os.path.dirname(os.path.abspath(__file__))

keyword = "profit"
categories_json = os.path.join(base_dir, '..', 'data/categorized_words.json')
document_path = os.path.join(base_dir, '..', 'data/copland_spellclean.txt')
model_name = os.path.join(base_dir, '..', 'data/fine-tuned-MacBERTh')
# model_name = model_name = os.path.join(base_dir, '..', 'data/emanjavacas/MacBERTh')
main(categories_json, document_path, model_name, keyword)
