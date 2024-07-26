'''This Python class turns cosine similarity outputted JSON (outputted from FindCosineSimilarityWords/ClusterUsingCosineSimilarity.py) into readable methods through customizable manipulation settings. This program also grabs the file metadata for each outputted JSON. This class could be used as a method to manipulate the cosine similarity results to create different interpretations.

This class contains different functions to decode: standardDecode, networkDecode, and different specialDecode functions. The specialDecode functions are meant for users to build their own program to decode the cosine similarity outputs.

Decoded self.outputJSONPath JSON file data type from def standardDecode():

{"fileName": {"Metadata": ["title String", "author String", year (as integer or list of integers], "baseword": ["returnedWord", score (float), "contextualSentence"], [["returnedWord(for duplicated words)", score (float), "contextualSentene"], ["returnedWord(for duplicated words)", score (float), "contextualSentence"]], ["returnedWord", score (float), "contextualSentence"]}, "fileName": ...}

Author: Jerry Zou'''

import json
class decodeCosSimOutput:
    def __init__(self, inputJSONPath, auxiliaryJSONPath, metadataJSONPath, outputJSONPath, batchWorkflowControl):
        self.inputJSONPath = inputJSONPath
        self.auxiliaryJSONPath = auxiliaryJSONPath
        self.metadataJSONPath = metadataJSONPath
        self.outputJSONPath = outputJSONPath
        self.batchWorkflowControl = batchWorkflowControl
        self.fileProcessCounter = 0 # This is specificallt for the funciton workflowControl to track how many files has been processed.

    
    def accessMetadata(self, fileName): #This helper function finds the metadata of a manuscript. Has to be a string input, such as "A00001"
        with open(self.metadataJSONPath, "r") as metadataFile:
            metadataContent = json.load(metadataFile)
        intendedMetadata = None
        for phase, fileNameDictionary in metadataContent.items():
            if fileName in fileNameDictionary:
                intendedMetadataDict = fileNameDictionary[fileName]
        intendedMetadata = [intendedMetadataDict["TITLE"], intendedMetadataDict["AUTHOR"], intendedMetadataDict["DATE"]]
        return intendedMetadata

    def workflowControl(self, resultDictionary): #workflow control, only for operations that need to dump into a json.
        if resultDictionary:
            try:
                with open(self.outputJSONPath, "r") as storageJSON:
                    existingDictionary = json.load(storageJSON)
            except (FileNotFoundError, json.JSONDecodeError):
                existingDictionary = {}

            existingDictionary.update(resultDictionary)

            with open(self.outputJSONPath, "w") as storageJSON:
                json.dump(existingDictionary, storageJSON, indent=4)
            self.fileProcessCounter += self.batchWorkflowControl
            print(f"Progress saved to JSON file. {self.fileProcessCounter} file cosine similiarity outputs processed.")

    def standardDecode(self):
        decodedDictionary = {}
        processedFilesCounter = 0
        with open(self.inputJSONPath, "r") as file:
            inputContentDictionary = json.load(file)
        for fileName, basewordDict in inputContentDictionary.items():
            decodedDictionary[fileName] = {}
            decodedDictionary[fileName]["Metadata"] = self.accessMetadata(fileName) # This uploads the metadata of the file into the dictionary

            for baseword, keywordMetadataDict in basewordDict.items():
                if baseword not in decodedDictionary[fileName]:
                    decodedDictionary[fileName][baseword] = []
                for keyword, metadata in keywordMetadataDict.items():
                    if len(metadata) > 1:
                        decodedDictionary[fileName][baseword].append([(keyword, score, sentence) for score, sentence, index in metadata])
                    elif len(metadata) == 1:
                        decodedDictionary[fileName][baseword].append((keyword, metadata[0][0], metadata[0][1])) # metadata[0][0] = CosSim score; metadata[0][1] = context sentence.

            for fileName, content in decodedDictionary.items():
                print(f"For file: {fileName}")
                for baseword, score in content.items():
                    print(f"For baseword {baseword}:")
                    for tuple in score:
                        print(tuple)

            processedFilesCounter += 1
            print(f"Processed {fileName}'s cosine similiarity output. {processedFilesCounter} out of {len(inputContentDictionary)} file outputs processed.")
            # if processedFilesCounter % self.batchWorkflowControl == 0:
            #     self.workflowControl(decodedDictionary)
            #     decodedDictionary.clear()
    
    # # This function looks at which returned word appeared most often in which categories and mauscripts. Basically, this function reverses th the structure of the dictionary where we have the returned word as the key. This function uses the JSON storage that has NOT been cleaned using the standard decode function above. Code revised from the file: Presentation Materials/CosSimVisualization/cosSimvVisualization.py
    def networkDecode(self):
        appearanceDictionary = {}
        with open(self.inputJSONPath, "r") as file:
            content = json.load(file)
        for filename, basewordDict in content.items():
            for baseword, keywordList in basewordDict.items():
                # print(type(keywordList))
                for keywordSelf, keywordInfo in keywordList.items():
                    cosSimScore = keywordInfo[0][0]
                    contextSentence = keywordInfo[0][1]
                    # print(keywordInfo[0][1])
                    if keywordSelf not in appearanceDictionary:
                        appearanceDictionary[keywordSelf] = {}
                    if baseword not in appearanceDictionary[keywordSelf]:
                        appearanceDictionary[keywordSelf][baseword] = [] #A list with the structure [filename, contextSentence, Title, Author, Year]

                    appearanceDictionary[keywordSelf][baseword].append(filename)
                    '''Comment or uncomment the following three lines with respect to whether the user needs to access the context sentence: '''
                    # appearanceDictionary[keywordSelf][baseword].append(contextSentence)
                    '''Comment or uncomment the following three lines with respect to whether the user needs to access metadata: '''
                    # for int in self.accessMetadata(filename):
                    #     appearanceDictionary[keywordSelf][baseword].append(int)
                    # print(f"Node processed word {keywordSelf}")

        counter = 0
        mostAppearingWord = []
        edgesCounterPerCategory = {}
        for returnedWord, appearance in appearanceDictionary.items():
            # print(returnedWord, appearance)
            if len(appearance) >= 2 and len(set(manuscriptList[0] for category, manuscriptList in appearance.items())) >= 2 : # This searches for words that appeared in more than X categories. The second conditional checks the number of manuscripts that the words appeared in.
                # mostAppearingWord.append((returnedWord, len(appearance))) #This is to update the most appearing words list that would be later on used for sorting.
                # counter +=1
                if "christ" in appearance:
                    counter += 1
                
                for category, file in appearance.items():
                    if category not in edgesCounterPerCategory:
                        edgesCounterPerCategory[category] = 0
                    edgesCounterPerCategory[category] += 1

                    if len(file) > 1:
                        for fileNum in file:

                            # metadataList = self.accessMetadata(fileNum)

                            counter += 1

                            # if metadataList[1] and metadataList[2]:
                            #     textInfo = fileNum + ";" + metadataList[0][0][:-10] + ";" + metadataList[1][0] + ";" + str(metadataList[2][0])
                            #     print(returnedWord, "|", category, "|", textInfo)
                            #     counter +=1 
                            # else:
                            #     print(returnedWord, "|", category, "|", fileNum)
                            #     counter +=1 

                    else:

                        # metadataList = self.accessMetadata(file[0])
                        counter +=1

                        # if metadataList[1] and metadataList[2]:
                        #     textInfo = file + ";" + metadataList[0][0][:-10] + ";" + metadataList[1][0] + ";" + str(metadataList[2][0])
                        #     print(returnedWord, "|", category, "|", textInfo)
                        #     counter +=1 
                        # else:
                        #     print(returnedWord, "|", category, "|", file, "Anonymous")
                        #     counter += 1

                print(returnedWord, len(appearance))
                print(returnedWord, appearance)
                
        print(counter)
        print(edgesCounterPerCategory)
        # mostAppearingWord = sorted(mostAppearingWord, key = lambda x: x[1], reverse=True)
        # print(f"Most appearing word: {mostAppearingWord}")

    def specialDecodeA(self): # This special decode function is used to test decoding one document within the dictionary at a time. Customize code as needed.
        decodedDictionary = {}
        with open(self.inputJSONPath, "r") as file:
            content = json.load(file)
        for fileName, basewordDict in content.items():
            decodedDictionary[fileName] = {}
            for baseword, keywordMetadataDict in basewordDict.items():
                if baseword not in decodedDictionary[fileName]:
                    decodedDictionary[fileName][baseword] = []
                for keyword, metadata in keywordMetadataDict.items():
                    if baseword != "christ" and baseword != "christs" and baseword != "light":
                        if len(metadata) > 1: 
                            decodedDictionary[fileName][baseword].append([(keyword, score, sentence) for score, sentence, index in metadata])
                        elif len (metadata) == 1:
                            decodedDictionary[fileName][baseword].append((keyword, metadata[0][0], metadata[0][1]))
        for fileName, content in decodedDictionary.items():
            if fileName == "A16864": #change according to need.

                print(f"For file: {fileName}")
                for baseword, score in content.items():
                    print(f"For baseword {baseword}:")
                    for tuple in score:
                        print(tuple)
    
    def specialDecodeB(self): #This special decode is meant for debugging or ther purposes where the developer needs to print out lists of the results for purposes like graphing or extracting specific information. This function only takes in JSONs that have been processed using standardDecode above. This function does not store information anywhere. It only prints to counsol.
        with open(self.inpoutJSOPath, "r") as file:
            inputJSONContent = json.load(file)
        # for 
        

if __name__ == "__main__":
    inputJSONPath = "/Users/Jerry/Desktop/combined300FilesOutput.json"
    auxiliaryJSONPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/FindCosineSimilarityWords/temporaryOutputforRepatedWords.json" #This does not do anything unless specified by adding customized code.
    metadataJSONPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/XMLProcessingAndTraining/ManuscriptMetadata/cleanedDocumentMetadata.json"
    outputJSONPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/FindCosineSimilarityWords/decodeCosSimOutput/output8SermonsReadable.json"
    batchWorkflowControl = 2 #workflow control. Processing this number of outputs each time before clearning memory of decoded dictionary.

    decodeClassObject = decodeCosSimOutput(inputJSONPath, auxiliaryJSONPath, metadataJSONPath, outputJSONPath, batchWorkflowControl)
    # decodeClassObject.standardDecode()
    decodeClassObject.networkDecode()
    # decodeClassObject.specialDecodeA()
    # decodeClassObject.specialDecodeB()