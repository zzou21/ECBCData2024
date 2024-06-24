import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

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
        text.lower()
    
    # Tokenize the text into sentences using nltk
    sentences = sent_tokenize(text)

    # Further split the sentences if there are more than 450 words
    split_sentences = []
    for sentence in sentences:
        words = sentence.split()
        while len(words) > 450:
            split_sentences.append(' '.join(words[:450]))
            words = words[450:]
        split_sentences.append(' '.join(words))

    # Strip whitespace and filter out empty sentences
    final_sentences = [s.strip() for s in split_sentences if s.strip()]
    
    return final_sentences

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
def compute_category_embeddings(categories, embeddings, tokenizer, model):
    categories_embeddings = {}
    for category, words in categories.items():
        category_embeddings = []
        for word in words:
            if word in embeddings:
                embedding = embeddings[word]
                category_embeddings.append(embedding)
            else:
                term_embedding = get_single_embedding(word, tokenizer, model)
        if category_embeddings:
            categories_embeddings[category] = np.mean(category_embeddings, axis=0)
        else:
            print(f"Warning: No embeddings found for category '{category}'.")
    return categories_embeddings

# Construct bias axes
def construct_bias_axes(category_embeddings):
    faith_bias_axis = category_embeddings["Faith"] - category_embeddings["Money"]
    desire_bias_axis = category_embeddings["Attraction"] - category_embeddings["Repulsion"]
    return faith_bias_axis, desire_bias_axis

# Function to project a word onto bias axes
def project_onto_bias_axis(word, embeddings, bias_axis):
    if word in embeddings:
        embedding = embeddings[word]
        projection = np.dot(embedding, bias_axis.T) / np.linalg.norm(bias_axis)
        return projection
    else:
        print(f"Error: Word '{word}' not found in embeddings.")
        return None

# Main function
def main(categories_json, document_path, model_name, keyword):
    # Load categories
    categories = load_categories(categories_json)
    
    # Load the MacBERTh model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Read and tokenize the document
    sentences = read_sentence_document(document_path)

    print(f"The maximum sentence length is: {len(max(sentences, key=len))} characters")

    embeddings = get_word_embedding(sentences, tokenizer, model)
    
    # Compute category embeddings
    category_embeddings = compute_category_embeddings(categories, embeddings, tokenizer, model)
    
    # Construct bias axes
    faith_bias_axis, desire_bias_axis = construct_bias_axes(category_embeddings)
    
    # Example: Project words from the document onto bias axes
    projection_faith = project_onto_bias_axis(keyword, embeddings, faith_bias_axis)
    projection_desire = project_onto_bias_axis(keyword, embeddings, desire_bias_axis)

    if (projection_faith is not None) and (projection_desire is not None):
        print(f"({projection_faith}, {projection_desire})")

# Now this is the main; feel free to change the following directory where fit
base_dir = os.path.dirname(os.path.abspath(__file__))

keyword = "profit"
categories_json = "categorized_words.json"
model_name = "finetuned_MacBERTh_Bible"
# model_name = model_name = os.path.join(base_dir, '..', 'data/emanjavacas/MacBERTh')

document_directory = os.path.join(base_dir, '..', 'EEBOphase2_1590-1639_body_texts')

for file_name in os.listdir(document_directory):
    document_path = os.path.join(document_directory, file_name)
    
    # Ensure it's a file (not a subdirectory)
    if os.path.isfile(document_path):
        # Call the main function with the document path and other arguments
        main(categories_json, document_path, model_name, keyword)