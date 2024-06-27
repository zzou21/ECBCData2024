import os
import torch
import numpy as np
import nltk
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.cluster import KMeans
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

def getWordEmbeddings(sentences, sentence_word_tokens, tokenizer, model):
    wordEmbeddings = {}
    for sentence, word_tokens in zip(sentences, sentence_word_tokens):
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0).numpy()  # Squeeze the first dimension
        token_ids = inputs['input_ids'].squeeze(0).numpy()  # Convert to numpy array

        for idx, token_id in enumerate(token_ids):
            word_token = tokenizer.convert_ids_to_tokens([token_id])[0]  # Convert token_id to a list and get the token
            if word_token not in stopwords and word_token.isalpha():
                unique_key = f"{word_token}_{len(wordEmbeddings)}"
                wordEmbeddings[unique_key] = embeddings[idx]  # Ensure this is a numpy array
    return wordEmbeddings

def getContextualEmbeddings(keywords, sentences, tokenizer, model):
    keyword_embeddings = {}
    for keyword, sentence in zip(keywords, sentences):
        print(f"Processing keyword '{keyword}' in sentence: {sentence}")  # Debug statement
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0).numpy()  # Squeeze the first dimension
        token_ids = inputs['input_ids'].squeeze(0).numpy()  # Convert to numpy array

        subwords = tokenizer.convert_ids_to_tokens(token_ids.tolist())
        print(f"Tokenized sentence: {subwords}")  # Debug statement

        for idx, subword in enumerate(subwords):
            if subword.lstrip("##") == keyword:
                unique_key = f"{keyword}_{len(keyword_embeddings)}"
                keyword_embeddings[unique_key] = embeddings[idx]  # Ensure this is a numpy array
                print(f"Found keyword '{keyword}' as subword '{subword}' at index {idx}")  # Debug statement
    print(f"Keyword embeddings: {keyword_embeddings}")  # Debug statement
    return keyword_embeddings

def calculate_keyword_similarity(keyword_embeddings):
    if len(keyword_embeddings) < 2:
        raise ValueError("Need at least two keywords to calculate similarity")
    keywords = list(keyword_embeddings.keys())
    embedding1 = np.array(keyword_embeddings[keywords[0]]).reshape(1, -1)  # Convert to numpy array and reshape to 2D
    embedding2 = np.array(keyword_embeddings[keywords[1]]).reshape(1, -1)  # Convert to numpy array and reshape to 2D
    similarity = cosine_similarity(embedding1, embedding2)[0][0]  # Use 2D arrays directly
    print(f"Similarity between '{keywords[0]}' and '{keywords[1]}': {similarity}")
    return similarity

def documentScore(embeddings, keywords, keyword_embeddings):
    keywordScoresDictionary = {word: 0 for word in keywords}
    keywordCountDictionary = {word: 0 for word in keywords}

    for word, embedding in embeddings.items():
        embedding = np.array(embedding).reshape(1, -1)  # Convert to numpy array and reshape to 2D
        for keyword, keyword_embedding in keyword_embeddings.items():
            keyword_embedding = np.array(keyword_embedding).reshape(1, -1)  # Convert to numpy array and reshape to 2D
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

def wordClusteringSingleDocument(wordEmbeddingResults, clusterCount):
    wordsList = []
    embeddingResultList = []
    for word, embedVectors in wordEmbeddingResults.items():
        wordsList.append(word)
        embeddingResultList.append(embedVectors)
    embeddings = np.array(embeddingResultList)
    kmeans = KMeans(n_clusters=clusterCount, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    print(labels)
    return wordsList, embeddings, labels

if __name__ == "__main__":
    textPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/data/copland_spellclean.txt"
    encodingMethods = ["utf-8", "latin-1", "cp1252"]
    keywords = ["sin", "profit"]
    context_sentences = [
        "if in any of these they have offend be not God rod of mortality just upon they for their sin but now belove almighty God have gracious look upon you and your people in pass.",
        "southerly than japan do japan I say lie under the same latitude that Virginia do abound with all thing for profit and pleasure be one of the mighty and opulent empire."
    ]
    modelPath = "emanjavacas/MacBERTh"
    tokenizer = AutoTokenizer.from_pretrained(modelPath)
    model = AutoModel.from_pretrained(modelPath)
    kmeansClusterCount = 5

    # Get embeddings for the keywords in their context sentences
    keyword_embeddings = getContextualEmbeddings(keywords, context_sentences, tokenizer, model)

    # Calculate similarity between keywords
    if len(keyword_embeddings) >= 2:
        calculate_keyword_similarity(keyword_embeddings)

    # Load and tokenize the text
    sentenceTok, wordTok = loadCleanTokenize(textPath, encodingMethods)

    # Get embeddings for the entire document
    documentWordEmbedding = getWordEmbeddings(sentenceTok, wordTok, tokenizer, model)

    # Calculate document score with contextual embeddings
    documentScore(documentWordEmbedding, keywords, keyword_embeddings)

    # Perform word clustering
    wordList, embeddings, labels = wordClusteringSingleDocument(documentWordEmbedding, kmeansClusterCount)
