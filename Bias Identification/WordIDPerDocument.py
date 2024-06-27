'''This file creates a graph for a document in which each point represents one word

Author: Jerry Zou'''
import json, os, torch, numpy as np, nltk
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

stopwords = set(nltk.corpus.stopwords.words("english"))

def loadCleanTokenize(textPath, encodingMethods):
    for method in encodingMethods:
        try:
            with open(textPath, "r", encoding=method) as textFile:
                textContent = textFile.read()
                break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("Cannot read the content with any provided encoding methods.")
    textContent = textContent.lower()
    sentenceTokenization = sent_tokenize(textContent)

    splitSentence = []
    longSentenceCounter = 0
    for sentence in sentenceTokenization:
        wordsInSentence = sentence.split(" ")
        while len(wordsInSentence) > 500:
            splitSentence.append(" ".join(wordsInSentence[:500]))
            wordsInSentence = wordsInSentence[500:]
            longSentenceCounter += 1
        splitSentence.append(" ".join(wordsInSentence))

    wordTokenizationAfterSentence = [word_tokenize(sentence) for sentence in splitSentence]
    print(f"long sentences: {longSentenceCounter}")
    return splitSentence, wordTokenizationAfterSentence

#embedding for document
def getWordEmbeddings(sentences, sentence_word_tokens, tokenizer, model):
    wordEmbeddings = {}
    for sentence, word_tokens in zip(sentences, sentence_word_tokens):
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0).numpy()
        token_ids = inputs['input_ids'].squeeze(0).numpy()

        # Iterate over the token IDs and their embeddings
        for idx, token_id in enumerate(token_ids):
            word_token = tokenizer.convert_ids_to_tokens(token_id.item())
            if word_token not in stopwords and word_token.isalpha():
                if word_token not in wordEmbeddings:
                    unique_key = f"{word_token}_{len(wordEmbeddings)}"
                    wordEmbeddings[unique_key] = embeddings[idx]  # Ensure this is a numpy array

    # print(type(wordEmbeddings))
    return wordEmbeddings

#embedding for keywd
def keywordEmbedding(keywords, tokenizer, model):
    keyword_embeddings = {}
    for word in keywords:
        inputs = tokenizer(word, return_tensors='pt', padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        wordEmbedding = embeddings[0, 1, :].numpy()
        keyword_embeddings[word] = wordEmbedding
    return keyword_embeddings

def calculate_keyword_similarity(keyword_embeddings):
    if len(keyword_embeddings) < 2:
        raise ValueError("Need at least two keywords to calculate similarity")
    keywords = list(keyword_embeddings.keys())
    embedding1 = np.array(keyword_embeddings[keywords[0]]).reshape(1, -1)  # Convert to numpy array and reshape to 2D
    embedding2 = np.array(keyword_embeddings[keywords[1]]).reshape(1, -1)  # Convert to numpy array and reshape to 2D
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    print(f"Similarity between '{keywords[0]}' and '{keywords[1]}': {similarity}")
    return similarity

def documentScore(embeddings, keywords, keyword_embeddings):
    keywordScoresDictionary = {word: 0 for word in keywords}
    keywordCountDictionary = {word: 0 for word in keywords}

    for word, embedding in embeddings.items():
        embedding = embedding.reshape(1, -1)  # Reshape to 2D
        for keyword, keyword_embedding in keyword_embeddings.items():
            keyword_embedding = keyword_embedding.reshape(1, -1)  # Reshape to 2D
            print(f"keyword_embedding in score: {keyword_embedding}")
            similarity = cosine_similarity(embedding, keyword_embedding)[0][0]
            if -1 <= similarity <= 1:  # Ensure the cosine similarity is within valid range
                angular_distance = np.arccos(similarity)
                keywordScoresDictionary[keyword.split('_')[0]] += angular_distance
                keywordCountDictionary[keyword.split('_')[0]] += 1

    print(f"unnormalized (sum of angular distances): {keywordScoresDictionary}")
    normalizedScores = {keyword: keywordScoresDictionary[keyword] / keywordCountDictionary[keyword] if keywordCountDictionary[keyword] > 0 else float('inf') for keyword in keywords}
    cosineNormalizedScores = {keyword: np.cos(normalizedScores[keyword]) for keyword in normalizedScores}
    print(f"normalized (cosine of average angular distance): {cosineNormalizedScores}")

    return cosineNormalizedScores

# def documentScore(embeddings, keywords, keyword_embeddings):
#     keywordScoresDictionary = {word: 0 for word in keywords}
#     keywordCountDictionary = {word: 0 for word in keywords}

#     for word, embeddingNumbersList in embeddings.items():
#         for embedding in embeddingNumbersList:
#             for keyword, keyword_embedding_list in keyword_embeddings.items():
#                 for keyword_embedding in keyword_embedding_list:
#                     similarity = cosine_similarity([embedding], [keyword_embedding])[0][0]
#                     if -1 <= similarity <= 1:  # Ensure the cosine similarity is within valid range
#                         angular_distance = np.arccos(similarity)
#                         keywordScoresDictionary[keyword] += angular_distance
#                         keywordCountDictionary[keyword] += 1

#     print(f"unnormalized (sum of angular distances): {keywordScoresDictionary}")
#     normalizedScores = {keyword: keywordScoresDictionary[keyword] / keywordCountDictionary[keyword] if keywordCountDictionary[keyword] > 0 else float('inf') for keyword in keywords}
#     cosineNormalizedScores = {keyword: np.cos(normalizedScores[keyword]) for keyword in normalizedScores}
#     print(f"normalized (cosine of average angular distance): {cosineNormalizedScores}")

# def documentScore(embeddings, keywords, tokenizer, model):
#     keywordEmbeddingNumbersDictionary = {word: keywordEmbedding(word, tokenizer, model) for word in keywords}
#     keywordScoresDictionary = {word: 0 for word in keywords}

#     # debug_info = {keyword: [] for keyword in keywords}  # FOR DEBUGGING

#     for word, embeddingNumbersList in embeddings.items():
#         for embedding in embeddingNumbersList:
#             for word, embeddingScore in keywordEmbeddingNumbersDictionary.items():
#                 similarity = cosine_similarity([embedding], [embeddingScore])[0][0]
#                 # similarity = np.dot(embedding, embeddingScore) / (np.linalg.norm(embedding) * np.linalg.norm(embeddingScore))
#                 keywordScoresDictionary[word] += np.arccos(similarity) #arc cosine

#                 # debug_info[word].append(similarity) # FOR DEBUGGING
#     print(f"unnormalized: {keywordScoresDictionary}")
#     numEmbeddingsSum = sum(len(embedding_list) for embedding_list in embeddings.values())
#     normalizedScores = {keyword: score / numEmbeddingsSum for keyword, score in keywordScoresDictionary.items()}
#     print(f"normalized: {np.cos(normalizedScores)}") #add cosine to normalizedScores

#     # for keyword, scores in debug_info.items(): # FOR DEBUGGING
#     #     print(f"Similarity scores for keyword '{keyword}': {scores[:10]}...")  # Print first 10 for brevity
#     #     print(f"Average similarity for keyword '{keyword}': {np.mean(scores)}")

#     return keywordScoresDictionary

def wordClusteringSingleDocument(wordEmbeddingResults, clusterCount):
    wordsList = []
    embeddingResultList = []
    for word, embedVectors in wordEmbeddingResults.items():
        for individualVector in embedVectors:
            wordsList.append(word)
            embeddingResultList.append(individualVector)
    embeddings = np.array(embeddingResultList)
    kmeans = KMeans(n_clusters=clusterCount, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    print(labels)
    return wordsList, embeddings, labels

def getContextualEmbeddings(keywords, sentences, tokenizer, model):
    keyword_embeddings = {}
    for keyword, sentence in zip(keywords, sentences):
        print(f"Processing keyword '{keyword}' in sentence: {sentence}")  # Debug statement
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0).numpy()
        token_ids = inputs['input_ids'].squeeze(0).numpy()

        subwords = tokenizer.convert_ids_to_tokens(token_ids.tolist())
        print(f"Tokenized sentence: {subwords}")  # Debug statement

        for idx, token_id in enumerate(token_ids):
            word_token = tokenizer.convert_ids_to_tokens(token_id.item())
            print(f"word token: {word_token}")
            if word_token == keyword or word_token.lstrip("##") == keyword:
                unique_key = f"{keyword}_{len(keyword_embeddings)}"
                # keyword_embeddings[unique_key] = embeddings[idx, :].numpy()
                keyword_embeddings[unique_key] = embeddings[idx]  # Ensure this is a numpy array

    print(f"Keyword embeddings: {keyword_embeddings}")  # Debug statement
    return keyword_embeddings


def calculate_similarity_between_contextual_words(embeddings1, embeddings2):
    similarities = []
    for emb1 in embeddings1:
        for emb2 in embeddings2:
            # similarity = cosine_similarity([emb1], [emb2])[0][0]
            similarity = cosine_similarity_manual(emb1, emb2)
            #arc cos
            similarities.append(similarity)
    return similarities

def cosine_similarity_manual(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)



if __name__ == "__main__":
    textPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/data/brinsley.txt"
    encodingMethods = ["utf-8", "latin-1", "cp1252"]
    keywords = ["london", "christ"]
    # ["money", "Christiandome", "Christian", "gold", "tobacco", "profit", "Christian Kingdom", "Bible", "stock", "shares", "school"]
    modelPath = "emanjavacas/MacBERTh"
    tokenizer = AutoTokenizer.from_pretrained(modelPath)
    model = AutoModel.from_pretrained(modelPath)
    kmeansClusterCount = 5

    sentenceKeyword = ["In this year of our Lord, fifteen hundred and eighty-three, the merchants of London did rejoice greatly, for the exchange of goods did bring forth much money, enriching the coffers of our fair city.", "In this year of grace, fifteen hundred and eighty-three, the humble folk of our village did kneel in prayer, seeking the mercy and blessings of Christ, the Redeemer of mankind."]

    categoryKeyword_embeddings = getContextualEmbeddings(keywords, sentenceKeyword, tokenizer, model)
    print(categoryKeyword_embeddings)
    calculate_keyword_similarity(categoryKeyword_embeddings)

    sentenceTok, wordTok = loadCleanTokenize(textPath, encodingMethods)
    documentWordEmbedding = getWordEmbeddings(sentenceTok, wordTok, tokenizer, model)
    # keywordEmbed = keywordEmbedding(keywords, tokenizer, model)

    # documentScore(documentWordEmbedding, keywords, tokenizer, model)
    documentScore(documentWordEmbedding, keywords, categoryKeyword_embeddings)

    wordList, embeddings, labels = wordClusteringSingleDocument(documentWordEmbedding, kmeansClusterCount)