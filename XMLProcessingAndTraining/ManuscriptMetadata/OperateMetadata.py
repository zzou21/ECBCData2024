import re, os, json

def findPublishingYear(jsonPath):
    with open(jsonPath, "r") as jsonContent:
        metadataContent = json.load(jsonContent)
    
    counter = 0  #could use this variable to count whichever number the user needs to track.

    for phase, filenameDictionary in metadataContent.items():
        for fileName, medatadaDictionary in filenameDictionary.items():
                                

if __name__ == "__main__":
    jsonPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/XMLProcessingAndTraining/ManuscriptMetadata/DocumentMetadata.json"
    findPublishingYear(jsonPath)