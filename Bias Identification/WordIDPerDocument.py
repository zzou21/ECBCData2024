'''This file creates a graph for a document in which each point represents one word

Author: Jerry Zou'''
import json, os, torch, numpy as np, nltk, matplotlib.pyplot as plt
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
        embeddings = outputs.last_hidden_state.squeeze(0)
        token_ids = inputs['input_ids'].squeeze(0)

        # Iterate over the token IDs and their embeddings
        for idx, token_id in enumerate(token_ids):
            word_token = tokenizer.convert_ids_to_tokens(token_id.item())
            if word_token not in stopwords and word_token.isalpha():
                if word_token not in wordEmbeddings:
                    wordEmbeddings[word_token] = []
                wordEmbeddings[word_token].append(embeddings[idx, :].numpy())
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
    embedding1 = keyword_embeddings[keywords[0]]
    embedding2 = keyword_embeddings[keywords[1]]
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    print(f"Similarity between '{keywords[0]}' and '{keywords[1]}': {similarity}")
    return similarity

def documentScore(embeddings, keywords, tokenizer, model):
    keywordEmbeddingNumbersDictionary = {word: keywordEmbedding(word, tokenizer, model) for word in keywords}
    keywordScoresDictionary = {word: 0 for word in keywords}

    # debug_info = {keyword: [] for keyword in keywords}  # FOR DEBUGGING

    for word, embeddingNumbersList in embeddings.items():
        for embedding in embeddingNumbersList:
            for word, embeddingScore in keywordEmbeddingNumbersDictionary.items():
                similarity = cosine_similarity([embedding], [embeddingScore])[0][0]
                # similarity = np.dot(embedding, embeddingScore) / (np.linalg.norm(embedding) * np.linalg.norm(embeddingScore))
                keywordScoresDictionary[word] += similarity

                # debug_info[word].append(similarity) # FOR DEBUGGING
    print(f"unnormalized: {keywordScoresDictionary}")
    numEmbeddingsSum = sum(len(embedding_list) for embedding_list in embeddings.values())
    normalizedScores = {keyword: score / numEmbeddingsSum for keyword, score in keywordScoresDictionary.items()}
    print(f"normalized: {normalizedScores}")

    # for keyword, scores in debug_info.items(): # FOR DEBUGGING
    #     print(f"Similarity scores for keyword '{keyword}': {scores[:10]}...")  # Print first 10 for brevity
    #     print(f"Average similarity for keyword '{keyword}': {np.mean(scores)}")

    return keywordScoresDictionary

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

def visualizeClusters(words, embeddings, labels):
    tsne = TSNE(n_components=2, random_state=42)
    final_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(14, 10))
    for i, word in enumerate(words):
        plt.scatter(final_embeddings[i, 0], final_embeddings[i, 1], c=f'C{labels[i]}')
        plt.annotate(word, (final_embeddings[i, 0], final_embeddings[i, 1]))

    plt.title("Word Embeddings Clustering")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()



#Calculating the cosine similarity of words within the context of the text
def getContextualEmbeddings(word, sentences, word_tokens, tokenizer, model):
    embeddings = []
    for sentence, tokens in zip(sentences, word_tokens):
        if word in tokens:
            print(f"Word '{word}' found in sentence: {sentence}")

            inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            sentence_embedding = outputs.last_hidden_state.squeeze(0)
            token_ids = inputs['input_ids'].squeeze(0)

            # Convert token ids back to tokens and find the matching subwords
            subwords = tokenizer.convert_ids_to_tokens(token_ids.tolist())
            for idx, subword in enumerate(subwords):
                # Check if subword matches the word we're looking for
                if subword == word or subword.lstrip("##") == word:
                    embeddings.append(sentence_embedding[idx].numpy())
                    break  # Consider only the first occurrence in the sentence
    return embeddings

def calculate_similarity_between_contextual_words(embeddings1, embeddings2):
    similarities = []
    for emb1 in embeddings1:
        for emb2 in embeddings2:
            # similarity = cosine_similarity([emb1], [emb2])[0][0]
            similarity = cosine_similarity_manual(emb1, emb2)
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
    keywords = ["Virginia", "Mars"]
    # ["money", "Christiandome", "Christian", "gold", "tobacco", "profit", "Christian Kingdom", "Bible", "stock", "shares", "school"]
    modelPath = "emanjavacas/MacBERTh"
    tokenizer = AutoTokenizer.from_pretrained(modelPath)
    model = AutoModel.from_pretrained(modelPath)
    kmeansClusterCount = 5

    sentenceTok, wordTok = loadCleanTokenize(textPath, encodingMethods)
    documentWordEmbedding = getWordEmbeddings(sentenceTok, wordTok, tokenizer, model)
    keywordEmbed = keywordEmbedding(keywords, tokenizer, model)
    calculate_keyword_similarity(keywordEmbed)

    documentScore(documentWordEmbedding, keywords, tokenizer, model)
    wordList, embeddings, labels = wordClusteringSingleDocument(documentWordEmbedding, kmeansClusterCount)
    
    # visualizeClusters(wordList, embeddings,labels)


    # money_embeddings = getContextualEmbeddings("posterity", sentenceTok, wordTok, tokenizer, model)
    # print(f"money embed: {money_embeddings}")
    # christendom_embeddings = getContextualEmbeddings("seminary", sentenceTok, wordTok, tokenizer, model)
    # print(f"christendom embed: {christendom_embeddings}")

    # # Calculate similarities between contextual embeddings
    # similarities = calculate_similarity_between_contextual_words(money_embeddings, christendom_embeddings)
    # print(similarities)