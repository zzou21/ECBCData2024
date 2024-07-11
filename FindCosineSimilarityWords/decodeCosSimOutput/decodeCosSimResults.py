'''This Python class turns cosine similarity outputted JSON (outputted from FindCosineSimilarityWords/ClusterUsingCosineSimilarity.py) into readable methods through customizable manipulation settings. This program also grabs the file metadata for each outputted JSON.

Decoded self.outputJSONPath JSON file data type:

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
    
    def accessMetadata(self, fileName): #This helper function finds the metadata of a manuscript. Has to be a string input, such as "A00001"
        with open(self.metadataJSONPath, "r") as metadataFile:
            metadataContent = json.load(metadataFile)
        intendedMetadata = None
        for phase, fileNameDictionary in metadataContent.items():
            if fileName in fileNameDictionary:
                intendedMetadataDict = fileNameDictionary[fileName]
        intendedMetadata = [intendedMetadataDict["TITLE"], intendedMetadataDict["AUTHOR"], intendedMetadataDict["DATE"]]
        return intendedMetadata

    def workflowControl(self, resultDictionary): #workflow control.
        fileProcessCounter = 0
        if resultDictionary:
            try:
                with open(self.outputJSONPath, "r") as storageJSON:
                    existingDictionary = json.load(storageJSON)
            except (FileNotFoundError, json.JSONDecodeError):
                existingDictionary = {}

            existingDictionary.update(resultDictionary)

            with open(self.outputJSONPath, "w") as storageJSON:
                json.dump(existingDictionary, storageJSON, indent=4)
            fileProcessCounter += self.batchWorkflowControl
            print(f"Progress saved to JSON file. {fileProcessCounter} file cosine similiarity outputs processed.")

    def standardDecode(self):
        decodedDictionary = {}
        processedFilesCounter = 0
        with open(self.inputJSONPath, "r") as file:
            content = json.load(file)
        for fileName, basewordDict in content.items():
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
            print(f"Processed {fileName}'s cosine similiarity output. {processedFilesCounter} out of {len(content)+1} file outputs processed.")
            if processedFilesCounter % self.batchWorkflowControl == 0:
                self.workflowControl(decodedDictionary)
                decodedDictionary.clear()

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
    
    def specialDecodeB(self):
        pass

if __name__ == "__main__":
    inputJSONPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/FindCosineSimilarityWords/temporaryOutputforRepatedWords.json"
    # decode(codedJSONPath)
    auxiliaryJSONPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/FindCosineSimilarityWords/temporaryOutputforRepatedWords.json"
    metadataJSONPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/XMLProcessingAndTraining/ManuscriptMetadata/cleanedDocumentMetadata.json"
    outputJSONPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/FindCosineSimilarityWords/decodeCosSimOutput/outputCosSimReadable.json"
    batchWorkflowControl = 2 #workflow control. Processing this number of outputs each time before clearning memory of decoded dictionary.

    decodeClassObject = decodeCosSimOutput(inputJSONPath, auxiliaryJSONPath, metadataJSONPath, outputJSONPath, batchWorkflowControl)
    decodeClassObject.standardDecode()
    # decodeClassObject.specialDecodeA()