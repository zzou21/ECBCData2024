'''This file will find the words surrounding a certain keywords to create a Cone-of-Words using the TF-IDF words identified in 2023. In the "comparison" function, it selects 20 of the top similar words (change the number to how many you'd like). The function creates a dictionary in the following format:

{"keyword": {"wordFromDocument": [similarityScore, "sentence where the word is from", index number of the sentence in the sentence tokenizeation variable of that file]}, "second word from document":[]}

This dictionary is then stored in a JSON file.

Author: Jerry Zou'''

from transformers import AutoTokenizer, AutoModel
import numpy as np, torch, nltk, json, heapq, os
# import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

class findConeOfWordsCommonKeywordDefinition:
    def __init__(self, filePath, folderPath, keywordJSON, storageJSON, model, tokenizer, returnTopWordsCount, specificFilesToAnalyze, batchWorkflowControl):
        self.filePath = filePath # A file path to the manuscript being examined
        self.folderPath = folderPath # A folder path that holds the individual TXTs to be scanned. Call either folderPath or filePath, not both.
        self.keywordJSON = keywordJSON # A path JSON file containing the keyword and the sentence it appears in
        self.model = model # MacBERTh
        self.tokenizer = tokenizer # MacBERTh tokenizer
        self.storageJSON = storageJSON
        self.returnTopWordsCount = returnTopWordsCount
        self.specificFilesToAnalyze = specificFilesToAnalyze
        self.batchWorkflowControl = batchWorkflowControl
        self.longSentenceCounter = 0 
        self.selectedKeywordsToUse = {} #This dictionary holds which specific parts of the keywordJSON that the user wants to use in finding cosine similarity words.
        self.keywordsUsed = []
        self.fileProcessCounter = 0 #This is to keep track of how many files have been processed while the program is running
        
    def processMainContent(self, contentPath):
        with open(contentPath, "r") as file:
            contentText = file.read()
        contentText = contentText.replace("\n", " ")
        tokenizedSentences = nltk.tokenize.sent_tokenize(contentText)
        tokenizedSentences = [sentence.strip() for sentence in tokenizedSentences if len(sentence.strip()) >= 30] # clear sentences that are too short to the point that it was mistakening tokenized or the tokenizer caught onto something uncessary.
        self.longSentenceCounter += sum(1 for sentence in tokenizedSentences if len(sentence) > 1900)
        return tokenizedSentences
    
    # def checkIfSentenceContainsAnyKeywords(self, sentence): #This function is used to optimize the program so we only need to embed the sentences that contain the keywords.
    #     for word in self.keywordsUsed:
    #         if word in sentence:
    #             return True
    #     return False
    
    def processEmbeddingMainContent(self, contentPath):
        wordEmbeddingsList = []
        sentenceToTokenize = self.processMainContent(contentPath)
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
    
    #Only run this interface when you area running on a local machine where you could engage with an interface.
    def selectKeywordToUseInterface(self):
        with open(self.keywordJSON, "r") as keywordFile:
            keywordContentDict = json.load(keywordFile)
        exitCommand = "#"
        userExitInput = ""
        keywordContentDictAsListOfTuple = [pair for pair in keywordContentDict.items()]
        selectedKeywords = []
        for i, pair in enumerate(keywordContentDictAsListOfTuple):
            print(f"Please type the number associated with the keyword to calculate: \n {i+1} --- {pair[0]}.")
        while userExitInput != exitCommand:
            inputCommand = input("Please type only the number of the keyword to add it to calculation queue. Type '#' to start calculation.  ")
            userExitInput = inputCommand
            if userExitInput != "#":
                try: inputCommandInt = int(inputCommand)
                except ValueError: print("Try again. Input numbers only.")
            selectedKeywords.append(keywordContentDictAsListOfTuple[inputCommandInt-1])
        self.selectedKeywordsToUse = {word: sentence for word, sentence in selectedKeywords}

    def processKeywords(self):
        keywordEmbeddings = []
        with open(self.keywordJSON, "r") as keywordContentFile:
            keywordContent = json.load(keywordContentFile)
            self.keywordsUsed = [keyword for keyword, sentence in keywordContent.items()]
        for keyword, sentence in keywordContent.items():
            sentenceToken = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**sentenceToken)
                tokenEmbeddings = outputs.last_hidden_state
            
            tokens = self.tokenizer.convert_ids_to_tokens(sentenceToken["input_ids"][0])
            currentKeyword = ""
            embeddingOfCurrentKeyword = []

            for token, embedding in zip(tokens, tokenEmbeddings[0]):
                if token == "[CLS]" or token == "[PAD]" or token == "[SEP]": continue
                if token.startswith("##"):
                    currentKeyword += token[2:]
                    embeddingOfCurrentKeyword.append(embedding)
                else:
                    if currentKeyword == keyword and embeddingOfCurrentKeyword:
                        keywordEmbeddings.append((keyword, torch.mean(torch.stack(embeddingOfCurrentKeyword), dim=0), sentence))
                    currentKeyword = token
                    embeddingOfCurrentKeyword = [embedding]
                
                if currentKeyword == keyword: embeddingOfCurrentKeyword.append(embedding)

            if currentKeyword == keyword and embeddingOfCurrentKeyword:
                keywordEmbeddings.append((keyword, torch.mean(torch.stack(embeddingOfCurrentKeyword), dim=0), sentence))

        return keywordEmbeddings

    # Unused: manually calculating cosine similarity
    def calculatingCosineSimilarity(self, v1, v2):
        v1 = v1.detach().numpy()
        v2 = v2.detach().numpy()
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)

    def comparison(self):
        with open(self.specificFilesToAnalyze, "r") as specificNames:
            specificFileNameList = json.load(specificNames)
        resultDictionary = {}
        processedFilesCounter = 0
        # filePathList = [os.path.join(self.folderPath, fileName) for fileName in os.listdir(self.folderPath) if fileName.endswith(".txt") and not fileName.startswith("._")]
        for fileName in os.listdir(self.folderPath):
            if fileName.endswith(".txt") and not fileName.startswith("._"):
                if fileName[:-4] in specificFileNameList.keys():
                    currentFilePath = os.path.join(self.folderPath, fileName)
                    keywordEmbeddingResults = self.processKeywords()
                    fileNameNoSuffix = fileName[:-4]
                    resultDictionary[fileNameNoSuffix] = {}
                    mainTextEmbeddingResults = self.processEmbeddingMainContent(currentFilePath)
                    for keyword, keywordCoordinates, keywordSentence in keywordEmbeddingResults:
                        similarities = []
                        for mainTextWord, mainTextWordCoordiantes, mainTextSentence, mainTextSentenceIndex in mainTextEmbeddingResults:
                            similarity = cosine_similarity(keywordCoordinates.unsqueeze(0), mainTextWordCoordiantes.unsqueeze(0)).item()
                            similarityTuple = (mainTextWord, similarity, mainTextSentence, mainTextSentenceIndex)
                            # print(f"similarityTuple : {similarityTuple}.")
                            if len(similarities) < self.returnTopWordsCount:
                                print(f"Processing word from text {fileNameNoSuffix}: {similarityTuple[0]}")
                                heapq.heappush(similarities, (similarityTuple[1], similarityTuple))
                            else:
                                heapq.heappushpop(similarities, (similarityTuple[1], similarityTuple))

                        similarities = [wordTuple for _, wordTuple in sorted(similarities, key=lambda x: x[1], reverse=True)]

                        for mainTextWord, similarityScore, mainTextSentence, mainTextSentenceIndex in similarities:
                            if keyword not in resultDictionary[fileNameNoSuffix]:
                                resultDictionary[fileNameNoSuffix][keyword] = {}
                            if mainTextWord not in resultDictionary[fileNameNoSuffix][keyword]: # make sure to remove christ and christs
                                resultDictionary[fileNameNoSuffix][keyword][mainTextWord] = []
                            resultDictionary[fileNameNoSuffix][keyword][mainTextWord].append((similarityScore, mainTextSentence, mainTextSentenceIndex))
                            if len(resultDictionary[fileNameNoSuffix][keyword]) > self.returnTopWordsCount: break

                    processedFilesCounter += 1
                    # print(f"Processed {len(resultDictionary)} files so far.")
                    if processedFilesCounter % self.batchWorkflowControl == 0:
                        self.workflowControl(resultDictionary)
                        resultDictionary.clear()

                else:
                    print(f"No files match the designated list of file names: {fileName[:-4]}")
        self.workflowControl(resultDictionary)
    
    def workflowControl(self, resultDictionary):
        if resultDictionary:
            try:
                with open(self.storageJSON, "r") as storageJSON:
                    existingDictionary = json.load(storageJSON)
            except (FileNotFoundError, json.JSONDecodeError):
                existingDictionary = {}

            existingDictionary.update(resultDictionary)

            with open(self.storageJSON, "w") as storageJSON:
                json.dump(existingDictionary, storageJSON, indent=4)
            self.fileProcessCounter += self.batchWorkflowControl
            print(f"Progress saved to JSON file. {self.fileProcessCounter} files processed.")

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("emanjavacas/MacBERTh")
    model = AutoModel.from_pretrained("emanjavacas/MacBERTh")
    
    filePath = "" #Ignore this variable for now#file path to the text to perform cosine similarity analysis.
    folderPath = "/hpc/group/datap2023ecbc/Team2024" #folder path to a folder that holds numerous TXT files to be scanned. Use either filePath or folderPath, not both.
    keywordJSONPath = "KeywordSentenceForBatch.json" #path to JSON file that stores the baseword and the contexual sentences.
    storageJSONPath = "batch3Storage.json" #path to JSON file that stores the output.
    specificFilesToAnalyze = "batch3.json"
    returnTopWordsCount = 60 # Number of output cosine similarity words you'd like to see.
    batchWorkflowControl = 2 # This is the workflow control methodology so that, when processing large amount of files, it won't be storing this number of files in the computer memory.

    findWordConeCommon = findConeOfWordsCommonKeywordDefinition(filePath, folderPath, keywordJSONPath, storageJSONPath, model, tokenizer, returnTopWordsCount, specificFilesToAnalyze, batchWorkflowControl) #initiates Python class object
    findWordConeCommon.comparison()
    # findWordCone.selectKeywordToUseInterface()