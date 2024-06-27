import json
import os
import statistics
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
        for embedding in embeddings[0]:
            word_embeddings.append(embedding.numpy())
    return word_embeddings

def get_word_embedding_with_word(chunk, tokenizer, model):
    word_embeddings = {}
    inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    for i, word in enumerate(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])):
        if word not in word_embeddings:
            word_embeddings[word] = embeddings[0, i, :].numpy()
    return word_embeddings

"""def get_single_embedding(word, tokenizer, model):
    # Tokenize the word
    inputs = tokenizer(word, return_tensors='pt')
    
    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the embedding for the word
    embeddings = outputs.last_hidden_state
    word_embedding = embeddings[0, 1, :].numpy()  # [CLS] token is at index 0, the word is at index 1
    return word_embedding"""

# Compute average vector for each category
def compute_category_embeddings(m,f, tokenizer, model):
    '''categories_embeddings = {}
    for category, words in categories.items():
        category_embeddings = []
        for word in words:
            term_embedding = get_single_embedding(word, tokenizer, model)
            category_embeddings.append(term_embedding)
        categories_embeddings[category] = np.mean(category_embeddings, axis=0)
    return categories_embeddings'''
    categories_embedding = {}
    money_dic = get_word_embedding_with_word(m, tokenizer, model)
    print('money' in money_dic)
    money = money_dic["money"]
    faith_dic = get_word_embedding_with_word(f, tokenizer, model)
    print('christ' in faith_dic)
    faith = faith_dic["christ"]
    categories_embedding["Money"] = money
    categories_embedding["Faith"] = faith
    return categories_embedding


# Construct bias axes
def construct_bias_axes(category_embeddings):
    faith_bias_axis = category_embeddings["Faith"] - category_embeddings["Money"]
    f = category_embeddings["Faith"]
    m = category_embeddings["Money"]

    dot = np.dot(f.T, m)
    print(dot / (np.linalg.norm(f) * np.linalg.norm(m)))
    return faith_bias_axis

# Function to project a word onto bias axes
def project_onto_bias_axis(embeddings, bias_axis):
    projections = []
    for vector in embeddings:
            projection = np.dot(vector, bias_axis.T) / np.linalg.norm(bias_axis)
            projections.append(projection)
    return projections

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
    # category_embeddings = compute_category_embeddings(categories, embeddings, tokenizer, model)
    money_sent = "In this year of our Lord, fifteen hundred and eighty-three, the merchants of London did rejoice greatly, for the exchange of goods did bring forth much money, enriching the coffers of our fair city."
    faith_sent = "In this year of grace, fifteen hundred and eighty-three, the humble folk of our village did kneel in prayer, seeking the mercy and blessings of Christ, the Redeemer of mankind."
    category_embeddings = compute_category_embeddings(money_sent, faith_sent, tokenizer, model)

    # Construct bias axes
    faith_bias_axis = construct_bias_axes(category_embeddings)
    
    # Example: Project words from the document onto bias axes
    projection_faith = project_onto_bias_axis(embeddings, faith_bias_axis)

    if (projection_faith is not None):
        print(f"{os.path.basename(document_path)}: ({statistics.median(projection_faith)})\n")
    

# Now this is the main; feel free to change the following directory where fit
base_dir = os.getcwd()

categories_json = os.path.join(base_dir, 'data/categorized_words.json')

# model_name = os.path.join(base_dir, 'data/fine-tuned-MacBERTh')
model_name = 'emanjavacas/MacBERTh'


document_directory = os.path.join(base_dir, 'data/copland_spellclean.txt')

main(categories_json, document_directory, model_name)