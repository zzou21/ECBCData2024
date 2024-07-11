'''This program overlays the metadata of each individual file names onto the calculation outputs from bias axes, cosine similarity, or other types of calculations. This file shares some common functions with "projectionTXTtoCSV.py".

This clas only takes in the file name format: "A00001"

Author: Jerry Zou'''

import json

class findMetadataForFile:
    def __init__ (self, nameToFind, metadataJSON):
        self.nameToFind = nameToFind # A string
        self.metadataJSON = metadataJSON #metadata JSON

    def accessMetadataJSON(self):
        with open(self.metadataJSON, "r") as jsonFile: content = json.load(jsonFile)
        return content

    def accessSpecificMetadata(self):
        metadataContent = self.accessMetadataJSON() # A dictionary, since it is directly returned from a JSON
        intendedMetadataDict = None
        for phase, fileNameDictionary in metadataContent.items():
            if self.nameToFind in fileNameDictionary:
                intendedMetadataDict = fileNameDictionary[self.nameToFind]
        return intendedMetadataDict
    
if __name__ == "__main__":
    nameToFind = "A00001"
    metadataJSON = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/XMLProcessingAndTraining/ManuscriptMetadata/cleanedDocumentMetadata.json"
    
    findMetadata = findMetadataForFile(nameToFind, metadataJSON)
    print(findMetadata.accessSpecificMetadata())