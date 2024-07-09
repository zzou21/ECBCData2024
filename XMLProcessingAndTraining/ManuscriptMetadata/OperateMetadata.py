'''This python class object cleans the document metadata json file. Each function in the class cleans one specific aspect of the json.

Authors: Jerry Zou and Lucas Ma'''
import re, os, json, string

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
                        for i, year in enumerate(metadataDictionary["DATE"]):
                            updatedYear = re.findall(r'\b\d{4}\b', year)
                            if updatedYear:
                                metadataDictionary["DATE"][i] = int(updatedYear[0])
                            else:
                                metadataDictionary["DATE"][i] = ""
                    elif len(metadataDictionary["DATE"]) == 1:
                        updatedYear = re.findall(r'\b\d{4}\b', metadataDictionary["DATE"][0])
                        if updatedYear: metadataDictionary["DATE"] = [int(updatedYear[0])]
                    elif not metadataDictionary["DATE"]:
                        metadataDictionary["DATE"] = ["Year unknown"]
        with open(self.inputFullMetadataJSONPath, "w") as updateJSON:
            json.dump(metadataContent, updateJSON, indent=4)

    def cleanPublicationCity(self):
        with open(self.inputFullMetadataJSONPath, "r") as jsonContent:
            metadataContent = json.load(jsonContent)
        for phase, filenameDictionary in metadataContent.items():
            for fileName, metadataDictionary in filenameDictionary.items():
                if metadataDictionary["PUBPLACE"]:
                    tempPubplaceHolder = metadataDictionary["PUBPLACE"][0]
                    translator = str.maketrans('', '', string.punctuation)
                    tempPubplaceHolder = tempPubplaceHolder.translate(translator)
                    toReplace = ["Printed at ", "Imprinted at ", "printed", "At "]
                    for replacePhrase in toReplace:
                        if replacePhrase in tempPubplaceHolder: tempPubplaceHolder = tempPubplaceHolder.replace(replacePhrase, "")
                    metadataDictionary["PUBPLACE"] = [tempPubplaceHolder]
                    print(metadataDictionary["PUBPLACE"])
                else: metadataDictionary["PUBPLACE"] = []
        with open(self.inputFullMetadataJSONPath, "w") as updateJSON:
            json.dump(metadataContent, updateJSON, indent=4)

    def cleanPublisher(self):
        with open(self.inputFullMetadataJSONPath, "r") as jsonContent:
            metadataContent = json.load(jsonContent)
        for phase, filenameDictionary in metadataContent.items():
            for fileName, metadataDictionary in filenameDictionary.items():
                if metadataDictionary["PUBLISHER"]:
                    metadataDictionary["PUBLISHER"] = metadataDictionary["PUBLISHER"]
                else: metadataDictionary["PUBLISHER"] = []
        with open(self.inputFullMetadataJSONPath, "w") as updateJSON:
            json.dump(metadataContent, updateJSON, indent=4)


    def cleanTitle(self):
        with open(self.inputFullMetadataJSONPath, "r") as jsonContent:
            metadataContent = json.load(jsonContent)
        for phase, filenameDictionary in metadataContent.items():
            for fileName, metadataDictionary in filenameDictionary.items():
                if metadataDictionary["TITLE"]:
                    metadataDictionary["TITLE"] = metadataDictionary["TITLE"]
                else: metadataDictionary["TITLE"] = []
        with open(self.inputFullMetadataJSONPath, "w") as updateJSON:
            json.dump(metadataContent, updateJSON, indent=4)

    def cleanAuthor(self):
        with open(self.inputFullMetadataJSONPath, "r") as jsonContent:
            metadataContent = json.load(jsonContent)
        for phase, filenameDictionary in metadataContent.items():
            for fileName, metadataDictionary in filenameDictionary.items():
                if metadataDictionary["AUTHOR"]:
                    newAuthorList = []

                    for author in metadataDictionary["AUTHOR"]:
                        name_pattern = re.compile(r'^([^,]+, [^,]+)')
                        match = name_pattern.match(author)
    
                        if match:
                            name_part = match.group(0)
                        else:
                            name_part = author

                        fl_index = name_part.find(' fl. ')
                        if fl_index != -1:
                            # Return the part of the entry before ".fl"
                            cleaned_entry = name_part[:fl_index].strip()
                            name_part = cleaned_entry

                        d_index = name_part.find(', d.')
                        if d_index != -1:
                            cleaned_entry = name_part[:d_index].strip()
                            name_part = cleaned_entry

                        if name_part[len(name_part)-1]==",":
                            name_part = name_part[:len(name_part)-1]

                        with open(os.path.join(currentDir, "test.txt"), "a") as f:
                            print(name_part, file=f)

                        newAuthorList.append(name_part)

                    metadataDictionary["AUTHOR"] = newAuthorList
        with open(self.inputFullMetadataJSONPath, "w") as updateJSON:
            json.dump(metadataContent, updateJSON, indent=4)

    def cleanYearAgain(self):
        with open(self.inputFullMetadataJSONPath, "r") as jsonContent:
            metadataContent = json.load(jsonContent)
        for phase, filenameDictionary in metadataContent.items():
            for fileName, metadataDictionary in filenameDictionary.items():
                if metadataDictionary["DATE"]:
                    if isinstance(metadataDictionary["DATE"], list):
                        metadataDictionary["DATE"] = [metadataDictionary["DATE"][i] for i in range(len(metadataDictionary["DATE"])) if isinstance(metadataDictionary["DATE"][i], int)]
                    elif isinstance(metadataDictionary["DATE"], int):
                        metadataDictionary["DATE"] = [metadataDictionary["DATE"]]
                    elif metadataDictionary["DATE"] == "":
                        metadataDictionary["DATE"] = []
        with open(self.inputFullMetadataJSONPath, "w") as updateJSON:
            json.dump(metadataContent, updateJSON, indent=4)
    
    def cleanFileName(self):
        with open(self.inputFullMetadataJSONPath, "r") as jsonContent:
            metadataContent = json.load(jsonContent)
        for phase, filenameDictionary in metadataContent.items():
            newFileNameMetadataContent = {}

            for fileName, metadataDictionary in filenameDictionary.items():
                if fileName.endswith(".P4"):
                    newFileName = fileName[:-3]
                else:
                    newFileName = fileName
                newFileNameMetadataContent[newFileName] = metadataDictionary
            metadataContent[phase] = newFileNameMetadataContent
        with open(self.inputFullMetadataJSONPath, "w") as updateJSON:
            json.dump(metadataContent, updateJSON, indent=4)

if __name__ == "__main__":
    currentDir = os.getcwd()
    inputFullMetadataJSONPath = os.path.join(currentDir, "XMLProcessingAndTraining/ManuscriptMetadata/cleanedDocumentMetadata.json")
    # inputFullMetadataJSONPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/XMLProcessingAndTraining/ManuscriptMetadata/Clean_metadata.json"
    metadataCleaningObject = metadataCleaning(inputFullMetadataJSONPath)
    
    '''Uncomment the functions you need to use below'''
    # metadataCleaningObject.cleanPublishingYear()
    # metadataCleaningObject.cleanPublicationCity()
    # metadataCleaningObject.cleanPublisher()
    # metadataCleaningObject.cleanAuthor()
    # metadataCleaningObject.cleanTitle()
    # metadataCleaningObject.cleanYearAgain()
    # metadataCleaningObject.cleanFileName()