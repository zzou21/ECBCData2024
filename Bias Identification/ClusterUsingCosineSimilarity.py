'''This file will find the words surrounding a certain keywords to create a Cone-of-Words using the TF-IDF words identified in 2023.

Author: Jerry Zou'''

from transformers import AutoTokenizer, AutoModel
# import pandas as pd
import numpy as np, torch, nltk

class findConeOfWords:
    def __init__(self, filePath, keywordJSON, model, tokenizer):
        self.filePath = filePath # A file path to the manuscript being examined
        self.keywordJSON = keywordJSON # A path JSON file containing the keyword and the sentence it appears in
        self.model = model # MacBERTh
        self.tokenizer = tokenizer # MacBERTh tokenizer
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
        for sentence in sentenceToTokenize:
            sentenceTokens = self.tokenizer(sentence, return_tensors = "pt", padding = True, truncation = True, max_length = 512)
            # print(f"sentence tokens: {sentenceTokens}")
            with torch.no_grad():
                outputs = model(**sentenceTokens)
                tokenEmbeddings = outputs.last_hidden_state
        
            tokens = self.tokenizer.convert_ids_to_tokens(sentenceTokens['input_ids'][0])
            currentWord = ""
            embeddingOfCurrentWord = []
            for token, embedding in zip(tokens, tokenEmbeddings[0]):
                if token == "[PAD]" or token == "[CLS]" or token == "[SEP]":
                    continue
                if token.startswith("##"):
                    currentWord += token[2:]
                else:
                    if currentWord and embeddingOfCurrentWord:
                        wordEmbeddingsList.append((currentWord, torch.mean(torch.stack(embeddingOfCurrentWord), dim=0)))
                    currentWord = token
                    embeddingOfCurrentWord = []
                embeddingOfCurrentWord.append(embedding)
            if currentWord:
                wordEmbeddingsList.append((currentWord, torch.mean(torch.stack(embeddingOfCurrentWord), dim=0)))
        
        return wordEmbeddingsList

    def cosine_similarity_manual(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("emanjavacas/MacBERTh")
    model = AutoModel.from_pretrained("emanjavacas/MacBERTh")
    filePath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/data/copland_spellclean.txt"
    keywordJSONPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/data/OneLevelKeywordSentence.json"
    findWordCone = findConeOfWords(filePath, keywordJSONPath, model, tokenizer)
    findWordCone.processEmbeddingMainContent()
    print(findWordCone.longSentenceCounter)