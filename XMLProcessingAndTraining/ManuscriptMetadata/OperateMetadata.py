import re, os, json

class metadataCleaning:
    def __init__(self, mainMetadataJSON):
        self.mainMetadataJSON = mainMetadataJSON

    def findPublishingYear(jsonPath):
        with open(jsonPath, "r") as jsonContent:
            metadataContent = json.load(jsonContent)
        
        counter = 0  #could use this variable to count whichever number the user needs to track.
        hasDate = 0
        totalfiles = 0
        fileNameListBetween = {}
        

        for phase, filenameDictionary in metadataContent.items():
            for fileName, metadataDictionary in filenameDictionary.items():
                totalfiles += 1
                if metadataDictionary["DATE"]:
                    hasDate += 1

                    # print(type(metadataDictionary["DATE"]))
                    # if len(metadataDictionary["DATE"]) > 1:
                    #     print(fileName)
                    # print(metadataDictionary["DATE"][0])
                    numbers = re.findall(r'\b\d{4}\b', metadataDictionary["DATE"][0])
                    if numbers:

                        newNum = int(numbers[0])
                        if newNum >= 1590 and newNum <= 1639:
                            print("here")
                            counter += 1
                            print(fileName)
                            # fileNameListBetween.append(metadataDictionary[""])
                            fileNameListBetween[fileName[:6]] = 1
                            print("added")
        return fileNameListBetween


if __name__ == "__main__":
    currentDir = os.getcwd()
    jsonPath = os.path.join(currentDir, "/XMLProcessingAndTraining/ManuscriptMetadata/DocumentMetadata.json")
    outputJSONPath = os.path.join(currentDir, "/XMLProcessingAndTraining/ManuscriptMetadata/FilesBetween15901639.json")
    # with open(outputJSONPath, "w") as file:
    #     json.dump(returnDictionary, file, indent=4)
    metadataCleaningObject = metadataCleaning(jsonPath)