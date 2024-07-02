'''This file will find the words surrounding a certain keywords to create a Cone-of-Words using the TF-IDF words identified in 2023.

Author: Jerry Zou'''

from transformers import AutoTokenizer, AutoModel
import numpy as np, torch, nltk, json
# import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

class findConeOfWords:
    def __init__(self, filePath, keywordJSON, storageJSON, model, tokenizer):
        self.filePath = filePath # A file path to the manuscript being examined
        self.keywordJSON = keywordJSON # A path JSON file containing the keyword and the sentence it appears in
        self.model = model # MacBERTh
        self.tokenizer = tokenizer # MacBERTh tokenizer
        self.storageJSON = storageJSON
        self.longSentenceCounter = 0
        
    def processMainContent(self):
        with open(self.filePath, "r") as manuscriptContentFile:
            manuscriptContent = manuscriptContentFile.read()
        tokenizedSentences = nltk.tokenize.sent_tokenize(manuscriptContent)
        for sentence in tokenizedSentences:
            if len(sentence) < 30: # clear sentences that are too short to the point that it was mistakening tokenized or the tokenizer caught onto something uncessary.
                tokenizedSentences.remove(sentence)
            if len(sentence) > 1900:
                self.longSentenceCounter += 1
        return tokenizedSentences
    
    def processEmbeddingMainContent(self):
        wordEmbeddingsList = []
        sentenceToTokenize = self.processMainContent()
        for tracker, sentence in enumerate(sentenceToTokenize):
            sentenceTokens = self.tokenizer(sentence, return_tensors = "pt", padding = True, truncation = True, max_length = 512)
            # print(f"sentence tokens: {sentenceTokens}")
            with torch.no_grad():
                outputs = self.model(**sentenceTokens)
                tokenEmbeddings = outputs.last_hidden_state
        
            tokens = self.tokenizer.convert_ids_to_tokens(sentenceTokens['input_ids'][0])
            currentWord = ""
            embeddingOfCurrentWord = []
            for token, embedding in zip(tokens, tokenEmbeddings[0]):
                if token == "[PAD]" or token == "[CLS]" or token == "[SEP]":
                    continue
                if token.startswith("##"):
                    currentWord += token[2:]
                    embeddingOfCurrentWord.append(embedding)
                else:
                    if currentWord and embeddingOfCurrentWord:
                        wordEmbeddingsList.append((currentWord, torch.mean(torch.stack(embeddingOfCurrentWord), dim=0), sentence, tracker))
                    currentWord = token
                    embeddingOfCurrentWord = []
                embeddingOfCurrentWord.append(embedding)
            if currentWord:
                wordEmbeddingsList.append((currentWord, torch.mean(torch.stack(embeddingOfCurrentWord), dim=0), sentence, tracker))
        
        return wordEmbeddingsList
    
    def processKeywords(self):
        keywordEmbeddings = []
        with open(self.keywordJSON, "r") as keywordFile:
            keywordContent = json.load(keywordFile)
        
        for keyword, sentence in keywordContent.items():
            sentenceToken = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**sentenceToken)
                tokenEmbeddings = outputs.last_hidden_state
            
            tokens = self.tokenizer.convert_ids_to_tokens(sentenceToken["input_ids"][0])
            currentKeyword = ""
            embeddingOfCurrentKeyword = []

            for token, embedding in zip(tokens, tokenEmbeddings[0]):
                if token == "[CLS]" or token == "[PAD]" or token == "[SEP]":
                    continue
                if token.startswith("##"):
                    currentKeyword += token[2:]
                    embeddingOfCurrentKeyword.append(embedding)
                else:
                    if currentKeyword == keyword and embeddingOfCurrentKeyword:
                        keywordEmbeddings.append((keyword, torch.mean(torch.stack(embeddingOfCurrentKeyword), dim=0), sentence))
                    currentKeyword = token
                    embeddingOfCurrentKeyword = [embedding]
                
                if currentKeyword == keyword:
                    embeddingOfCurrentKeyword.append(embedding)

            if currentKeyword == keyword and embeddingOfCurrentKeyword:
                keywordEmbeddings.append((keyword, torch.mean(torch.stack(embeddingOfCurrentKeyword), dim=0), sentence))
        return keywordEmbeddings

    def calculatingCosineSimilarity(self, v1, v2):
        v1 = v1.detach().numpy()
        v2 = v2.detach().numpy()
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)
    
    def comparison(self):
        resultDictionary = {}
        keywordEmbeddingResults = self.processKeywords()
        mainTextEmbeddingResults = self.processEmbeddingMainContent()
        for keyword, keywordCoordinates, keywordSentence in keywordEmbeddingResults:
            similarities = []
            for mainTextWord, mainTextWordCoordiantes, mainTextSentence, mainTextSentenceIndex in mainTextEmbeddingResults:
                similarity = cosine_similarity(keywordCoordinates.unsqueeze(0), mainTextWordCoordiantes.unsqueeze(0)).item()

                # oneSimilarityScore = self.calculatingCosineSimilarity(keywordCoordinates, mainTextWordCoordiantes)
                similarities.append((mainTextWord, similarity, mainTextSentence, mainTextSentenceIndex))

            similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

            for mainTextWord, similarityScore, mainTextSentence, mainTextSentenceIndex in similarities:
                if keyword not in resultDictionary:
                    resultDictionary[keyword] = {}
                
                if mainTextWord not in resultDictionary[keyword]:
                    resultDictionary[keyword][mainTextWord] = []
                
                resultDictionary[keyword][mainTextWord].append((similarityScore, mainTextSentence, mainTextSentenceIndex))
                if len(resultDictionary[keyword]) > 20:
                    break
                    # print(f"For keyword {keyword}:")
                    # print(f"Word: {mainTextWord}, Similarity: {similarityScore}, Sentence: {mainTextSentence}")
        # print(resultDictionary)
        for keyword, mainTextWordDictionary in resultDictionary.items():
            for wordFromMainText, listMetadata in mainTextWordDictionary.items():
                if len(listMetadata) > 1:
                    print(f"For keyword {keyword}:")
                    for tupleMetadata in listMetadata:
                        print(f"Word: '{wordFromMainText}' has a similarity score of {tupleMetadata[0]} in sentence '{tupleMetadata[1]}'.")
                else:
                    print(f"For keyword {keyword}")
                    print(f"Word: '{wordFromMainText}' has a similarity score of {listMetadata[0][0]} in sentence '{listMetadata[0][1]}'.")
        with open(self.storageJSON, "w") as storageJSON:
            json.dump(resultDictionary, storageJSON, indent=4)
        

        # min_val = None
        # max_val = None
        # for elem in result:
        #     float_elem = float(elem[2]) 
        #     if min_val is None or float_elem < min_val: 
        #         min_val = float_elem 
        #     if max_val is None or float_elem > max_val: 
        #         max_val = float_elem
        # print(f"minimum score: {min_val}; maxinum score: {max_val}.")

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("emanjavacas/MacBERTh")
    model = AutoModel.from_pretrained("emanjavacas/MacBERTh")
    filePath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/data/copland_spellclean.txt"
    keywordJSONPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/data/OneLevelKeywordSentence.json"
    storageJSONPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/Bias Identification/CosSimWordClusterResult.json"
    findWordCone = findConeOfWords(filePath, keywordJSONPath, storageJSONPath, model, tokenizer)
    # print(findWordCone.processEmbeddingMainContent())
    findWordCone.comparison()