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
def get_word_embedding(chunks, tokenizer, model, word):
    word_times = {}
    word_embeddings = {}
    for chunk in chunks:
        
        in_set = False
        for token in chunk.split(" "):
            if (token in substitute_word(word, document_dir=os.getcwd())) or (token == word):
                in_set = True
        if not in_set:
            continue

        inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        for i, word in enumerate(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])):
            if word not in word_times:
                word_times[word] = 1
            else:
                word_times[word]+=1
            if word not in word_embeddings:
                word_embeddings[word] = embeddings[0, i, :].numpy()
            else:
                word_embeddings[word] = (word_embeddings[word] * (word_times[word]-1) + embeddings[0, i, :].numpy()) / word_times[word]
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

# Find the substitute words for the original keyword, by iterating over the standard word list
def substitute_word(word, document_dir):
    json_dir = os.path.join(document_dir, "standardizedwords.json")
    standardWord = load_categories(json_dir)
    ret = []
    for term, equals in standardWord.items():
        for spell in equals:
            if spell==word:
                ret.append(term)
    return ret

# Function to project a word onto bias axes
def project_onto_bias_axis(word, embeddings, bias_axis, document_dir):
    projection = 0
    if word in embeddings:
        embedding = embeddings[word]
        projection = np.dot(embedding, bias_axis.T) / np.linalg.norm(bias_axis)
    else:
        for substitute in substitute_word(word, document_dir):
            if substitute in embeddings:
                embedding = embeddings[substitute]
                projection = np.dot(embedding, bias_axis.T) / np.linalg.norm(bias_axis)
            else:
                projection = 0
    return projection

# Main function
def main(categories_json, document_path, model_name, keyword, document_directory):
    # Load categories
    categories = load_categories(categories_json)
    
    # Load the MacBERTh model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Read and tokenize the document
    sentences = read_sentence_document(document_path)
    embeddings = get_word_embedding(sentences, tokenizer, model, keyword)
    
    # Compute category embeddings
    category_embeddings = compute_category_embeddings(categories, tokenizer, model)

    # Construct bias axes
    faith_bias_axis = construct_bias_axes(category_embeddings)
    
    # Example: Project words from the document onto bias axes
    projection_faith = project_onto_bias_axis(keyword, embeddings, faith_bias_axis, document_directory)
    
    result_file = os.path.join(document_directory, "./data/projection_result.txt")
    if (projection_faith is not None):
        result = f"{os.path.basename(document_path)}: {projection_faith}\n"
        with open(result_file, 'a') as f:
            f.write(result)

# Now this is the main; feel free to change the following directory where fit
base_dir = os.getcwd()

keyword = "profit"
categories_json = os.path.join(base_dir, './data/categorized_words.json')
# model_name = "finetuned_MacBERTh_Bible"
model_name = model_name = os.path.join('emanjavacas/MacBERTh')

document_directory = os.path.join('/Users/lucasma/Downloads/EEBOphase2_1590-1639_body_texts')

result = load_categories(os.path.join(base_dir, "./data/output_projection.json"))

for file_name in os.listdir(document_directory):
    document_path = os.path.join(document_directory, file_name)
    if file_name in result:
        continue

    # Ensure it's a file (not a subdirectory)
    if os.path.isfile(document_path) and document_path[0] != ".":
        # Call the main function with the document path and other arguments
        main(categories_json, document_path, model_name, keyword, base_dir)