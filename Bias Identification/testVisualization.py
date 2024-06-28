import nltk
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("emanjavacas/MacBERTh")
model = AutoModel.from_pretrained("emanjavacas/MacBERTh")

def processMainText(text, selected_words):
    words_embeddings = []
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state
        
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        current_word = ""
        current_embeddings = []

        for token, embedding in zip(tokens, token_embeddings[0]):
            if token == "[PAD]" or token == "[CLS]" or token == "[SEP]":
                continue
            print(f"encoded token: {token}")

            if token.startswith("##"):
                current_word += token[2:]
            else:
                if current_word in selected_words:
                    words_embeddings.append((current_word, torch.mean(torch.stack(current_embeddings), dim=0)))
                current_word = token
                current_embeddings = []
            current_embeddings.append(embedding)

        if current_word in selected_words:
            words_embeddings.append((current_word, torch.mean(torch.stack(current_embeddings), dim=0)))
    
    return words_embeddings

def encodeKeyword(word, context_sentence):
    inputs = tokenizer(context_sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    current_word = ""
    current_embeddings = []

    for token, embedding in zip(tokens, token_embeddings[0]):
        if token == "[PAD]" or token == "[CLS]" or token == "[SEP]":
            continue

        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word == word:
                return torch.mean(torch.stack(current_embeddings), dim=0).numpy()
            current_word = token
            current_embeddings = []
        current_embeddings.append(embedding)

    if current_word == word:
        return torch.mean(torch.stack(current_embeddings), dim=0).numpy()
    return None

def cosine_similarity_manual(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

def project_onto_axis(v, axis):
    """
    Project vector v onto axis.
    """
    return np.dot(v, axis) / np.linalg.norm(axis)

if __name__ == "__main__":

    text_path = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/data/A00151.txt"
    with open(text_path, "r") as file:
        text_content = file.read()

    context_sentence = "I have be full persuade upon happy experience I trust that I can never employ my thought and travel more acceptable in any earthly thing or whereby a great benefit may redound both to church and commonwealth and in some sort to every soul then in search out and set forth to the view of all the short sure and most easy entrance to all good learning and how with certain hope of good success all may proceed therein who know not the grievous complaint which to the disgrace of learning be make almost in every place the usual complaint against non-proficiency in school for the injury do to country town parent and child because in so many school the child which be the chief hope of parent and posterity be either spoil altogether or else do profit so very little where good be do how hardly it be effect common and for the most part wherein any good be do that it be ordinary effect by the endless vexation of the painful master the extreme labour and ^errour of the poor scholar with endure far overmuch and long severity now whence proceed all this but because so few of those who undertake this function a chief cause hereof want of knowledge of a right course of teach be acquaint with any good method or right order of instruction fit for a grammar school this therefore have be in my heart to show my love and duty to all sort in seek for my part to deliver the poor painful and honest mind schoolmaster from this reproach and grief the author desire to help all this and to help withal to supply this so great a want and in stead hereof my earnest desire have be to procure a perpetual benefit to all estate and degree and to procure a perpetual benefit to all posterity even to every man for his child and posterity by endeavour to make the path to all good learning more even and please in the first entrance then former age have know and thereby also in the continual proceed afterward."
    keywords = ["disgrace", "grammar"]

    # Select the words from the document to analyze
    selected_words = ["christian", "labour"]

    # Process only the selected words
    mainTextEmbedding = processMainText(text_content, selected_words)

    # Process keywords
    keywordEmbedding = [(word, encodeKeyword(word, context_sentence)) for word in keywords]

    combinedEmbedding = mainTextEmbedding + keywordEmbedding

    # Create a DataFrame for combined embeddings
    embedding_data = {
        "word": [word for word, _ in combinedEmbedding],
        "embedding": [embedding for _, embedding in combinedEmbedding]
    }

    df = pd.DataFrame(embedding_data)

    # Extract keyword embeddings
    keyword_1_embedding = df[df['word'] == 'disgrace']['embedding'].values[0]
    keyword_2_embedding = df[df['word'] == 'grammar']['embedding'].values[0]
    axis = keyword_2_embedding - keyword_1_embedding

    # Calculate projection of each word onto the axis
    projections = df['embedding'].apply(lambda x: project_onto_axis(np.array(x), axis))

    # Print projections
    listOfProjection = []
    for word, projection in zip(df['word'], projections):
        listOfProjection.append(projection)
        print(f"Word: {word}, Projection on axis: {projection}")
    