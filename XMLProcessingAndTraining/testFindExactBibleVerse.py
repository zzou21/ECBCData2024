import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('/Users/Jerry/Desktop/ShuffleFullBibleModel')

# Load the pre-trained model from the saved folder
model_path = '/Users/Jerry/Desktop/ShuffleFullBibleModel'
tokenizer = BertTokenizer.from_pretrained(model_path)

# Load the model configuration and enable output_hidden_states
config = BertConfig.from_pretrained(model_path)
config.output_hidden_states = True  # Enable output_hidden_states

# Load the model with the modified configuration
model = BertForSequenceClassification.from_pretrained(model_path, config=config)

# Ensure the model is in evaluation mode
model.eval()

# Move the model to the appropriate device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Load the CSV file
df = pd.read_csv('/Users/Jerry/Desktop/FinalFullGenevaBibleRecognitionDataSetShuffled.csv')

# Filter out only the Bible verses
bible_verses_df = df[df['Label'] == 1]
bible_verses = bible_verses_df['Text'].tolist()

def get_sentence_embedding(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        # Take the mean of the last hidden state to get the sentence embedding
        hidden_states = outputs.hidden_states[-1]
        sentence_embedding = torch.mean(hidden_states, dim=1)
    return sentence_embedding.cpu().numpy()

# Generate embeddings for the verses
verse_embeddings = np.array([get_sentence_embedding(verse, model, tokenizer) for verse in bible_verses])

def identify_verse(input_text, model, tokenizer, verse_embeddings, bible_verses):
    input_embedding = get_sentence_embedding(input_text, model, tokenizer)
    
    # Classify if it is a Bible verse
    inputs = tokenizer([input_text], return_tensors='pt', truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        classification = model(**inputs).logits
    is_bible_verse = torch.argmax(classification, dim=-1).item()
    
    if is_bible_verse == 1:  # Check if the predicted label is 1, indicating a Bible verse
        # Compute similarity with all Bible verses
        similarities = cosine_similarity(input_embedding, verse_embeddings)
        # Find the closest verse
        closest_verse_index = np.argmax(similarities)
        return bible_verses[closest_verse_index]
    else:
        return "Not a Bible verse"

# Example usage
input_text = "Your input text here"
result = identify_verse(input_text, model, tokenizer, verse_embeddings, bible_verses)
print(result)