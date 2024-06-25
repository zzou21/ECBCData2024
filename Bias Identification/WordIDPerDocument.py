'''This file creates a graph for a document in which each point represents one word

Author: Jerry Zou'''
import json, os, torch, numpy as np, nltk
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.cluster import KMeans

modelPath = "emanjavacas/MacBERTh"
tokenizer = AutoTokenizer.from_pretrained(modelPath)
model = AutoModel.from_pretrained(modelPath)

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
        embeddings = outputs.last_hidden_state.squeeze(0)
        token_ids = inputs['input_ids'].squeeze(0)

        # Iterate over the token IDs and their embeddings
        for idx, token_id in enumerate(token_ids):
            word_token = tokenizer.convert_ids_to_tokens(token_id.item())
            if word_token not in stopwords and word_token.isalpha():
                if word_token not in wordEmbeddings:
                    wordEmbeddings[word_token] = []
                wordEmbeddings[word_token].append(embeddings[idx, :].numpy())
    print(type(wordEmbeddings))
    return wordEmbeddings

#embedding for keywd
def keywordEmbedding(word, tokenizer, model):
    inputs = tokenizer(word, return_tensors='pt', padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    wordEmbedding = embeddings[0, 1, :].numpy()
    print(type(wordEmbedding))
    return wordEmbedding

def documentScore(embeddings, keywords, tokenizer, model):
    keywordEmbeddingNumbersDictionary = {keywords: keywordEmbedding(word, tokenizer, model) for word in keywords}
    keywordScoresDictionary = {word: 0 for word in keywords}
    for word, embeddingNumbersList in embeddings.items():
        for embedding in embeddingNumbersList:
            for word, embeddingScore in keywordEmbeddingNumbersDictionary.items():
                similarity = np.dot(embedding, embeddingScore) / (np.linalg.norm(embedding) * np.linalg.norm(embeddingScore))
                keywordScoresDictionary[word] += similarity
    numEmbeddingsSum = sum(len(embedding_list) for embedding_list in embeddings.values())
    normalizedScores = {keyword: score / numEmbeddingsSum for keyword, score in keywordScoresDictionary.items()}

    return normalizedScores

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



if __name__ == "__main__":
    textPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/data/copland_spellclean.txt"
    encodingMethods = ["utf-8", "latin-1", "cp1252"]
    keywords = ["money", "Christiandome"]
    modelPath = "emanjavacas/MacBERTh"
    tokenizer = AutoTokenizer.from_pretrained(modelPath)
    model = AutoModel.from_pretrained(modelPath)

    sentenceTok, wordTok = loadCleanTokenize(textPath, encodingMethods)
    documentWordEmbedding = getWordEmbeddings(sentenceTok, wordTok, tokenizer, model)
    keywordEmbed = keywordEmbedding(keywords, tokenizer, model)