import json
import os, re
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Make sure to download the punkt tokenizer
nltk.download('punkt')

def get_sentence_embedding(sentence, tokenizer, model):
    word_embeddings = {}
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    for i, word in enumerate(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])):
        if word not in word_embeddings:
            word_embeddings[word] = embeddings[0, i, :].numpy()
    return word_embeddings

# Function to project a word onto bias axes
def project_onto_bias_axis(embedding, bias_axis):
    return np.dot(embedding, bias_axis.T) / np.linalg.norm(bias_axis)

# Main function
def main(sentences, model_name, keyword):
    # Load the MacBERTh model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Read and tokenize the document
    embedding_1 = get_sentence_embedding(sentences[0], tokenizer, model)[keyword]
    embedding_2 = get_sentence_embedding(sentences[1], tokenizer, model)[keyword]


    good_embedding = get_sentence_embedding("good virtues should be commended.", tokenizer, model)["good"]
    bad_embedding = get_sentence_embedding("We dislike bad people.", tokenizer, model)["bad"]
    
    axis = good_embedding - bad_embedding

    # Example: Project words from the document onto bias axes
    print(project_onto_bias_axis(embedding_1, axis))
    print(project_onto_bias_axis(embedding_2, axis))

# Now this is the main; feel free to change the following directory where fit
keyword = "fool"

sentences = ["You are such a fool.", "You are such an adorable fool."]

model_name = 'emanjavacas/MacBERTh'
# Call the main function with the document path and other arguments
main(sentences, model_name, keyword)
