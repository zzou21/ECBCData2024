import nltk
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


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



if __name__ == "__main__":

    text_path = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/data/brinsley.txt"
    with open(text_path, "r") as file:
        text_content = file.read()

    text = "I just came from a happy party in Baltimore. It was such a joyful event and everyone was happy."
    keywords = ["happy", "baltimore"]


    # Define the words to visualize
    selected_words = ["heaven", "profit"]  # Replace with actual words

    # Process only the selected words
    mainTextEmbedding = processMainText(text_content, selected_words)

    #process keyword:
    keywordEmbedding = [(word, encodeKeyword(word, text)) for word in keywords]

    combinedEmbedding = mainTextEmbedding + keywordEmbedding

    for word, embedding in combinedEmbedding:
        print(f"word: {word} with vectors: {embedding[:5]}...")





    # embedding_data = {
    #     "word": [word for word, _ in mainTextEmbedding],
    #     "embedding": [embedding.numpy() for _, embedding in mainTextEmbedding]
    # }

    # df = pd.DataFrame(embedding_data)

    # # Convert filtered embeddings to a numpy array
    # filtered_embeddings = np.stack(df['embedding'].values)

    # # Apply PCA to reduce to 3 dimensions
    # pca = PCA(n_components=3)
    # reduced_embeddings = pca.fit_transform(filtered_embeddings)

    # # Add the reduced dimensions to the filtered DataFrame
    # df['x'] = reduced_embeddings[:, 0]
    # df['y'] = reduced_embeddings[:, 1]
    # df['z'] = reduced_embeddings[:, 2]

    # # Create a 3D scatter plot
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(df['x'], df['y'], df['z'], c='blue', marker='o')

    # # Annotate points with their words
    # for i in range(len(df)):
    #     ax.text(df['x'][i], df['y'][i], df['z'][i], df['word'][i])

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # plt.show(block=True)  # Ensure the plot window remains open
