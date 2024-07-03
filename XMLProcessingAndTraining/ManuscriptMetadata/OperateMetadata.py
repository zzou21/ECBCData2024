import re, os, json

class metadataCleaning:
    def __init__(self, inputFullMetadataJSONPath):
        self.inputFullMetadataJSONPath = inputFullMetadataJSONPath

    def cleanPublishingYear(self):
        with open(self.inputFullMetadataJSONPath, "r") as jsonContent:
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
                    if len(metadataDictionary["DATE"]) > 1:
                        # print(metadataDictionary["DATE"])
                        for i, year in enumerate(metadataDictionary["DATE"]):
                            # print(type(year))
                            updatedYear = re.findall(r'\b\d{4}\b', year)
                            # print(updatedYear)
                            if updatedYear:
                                metadataDictionary["DATE"][i] = int(updatedYear[0])
                            else:
                                metadataDictionary["DATE"][i] = ""
                                
                        print(metadataDictionary["DATE"])

                    # print(metadataDictionary["DATE"])

                            
                    #     # print(metadataDictionary["DATE"])
                    # elif len(metadataDictionary["DATE"]) == 1:
                    #     metadataDictionary["DATE"][0] = re.findall(r'\b\d{4}\b', metadataDictionary["DATE"][0])
                    #     # print(metadataDictionary["DATE"][0])
                    # else:
                    #     pass

if __name__ == "__main__":
    # currentDir = os.getcwd()
    # jsonPath = os.path.join(currentDir, "/XMLProcessingAndTraining/ManuscriptMetadata/DocumentMetadata.json")
    inputFullMetadataJSONPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/XMLProcessingAndTraining/ManuscriptMetadata/DocumentMetadata.json"
    # outputJSONPath = os.path.join(currentDir, "/XMLProcessingAndTraining/ManuscriptMetadata/FilesBetween15901639.json")
    # with open(outputJSONPath, "w") as file:
    #     json.dump(returnDictionary, file, indent=4)
    metadataCleaningObject = metadataCleaning(inputFullMetadataJSONPath)
    metadataCleaningObject.cleanPublishingYear()



                    


                    # if numbers:
                    #     newNum = int(numbers[0])

                        # if newNum >= 1590 and newNum <= 1639:
                        #     print("here")
                        #     counter += 1
                        #     print(fileName)
                        #     # fileNameListBetween.append(metadataDictionary[""])
                        #     fileNameListBetween[fileName[:6]] = 1
                        #     print("added")