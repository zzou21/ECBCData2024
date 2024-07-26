'''This program searches through the metadata JSON and selects the metadata of files that the user needs and stores that in another JSON file.

Author: Jerry Zou'''
import json, csv, sys, os

class findSpecificMetadata:
    def __init__ (self, searchJSON, outputJSON, filesToSearch):
        self.searchJSON = searchJSON
        self.outputJSON = outputJSON
        self.filesToSearch = filesToSearch
        self.listOfFilesToSearch = []

    def findSpecificMeta(self):
        specificJsonDictionary = {}
        with open(self.searchJSON, "r") as searchFile:
            searchContent = json.load(searchFile)
        for version, contentDictionary in searchContent.items():
            specificJsonDictionary[version] = {}
            for fileName, metadataContent in contentDictionary.items():
                for targetFileName in self.listOfFilesToSearch:
                    if targetFileName in fileName:
                        specificJsonDictionary[version][fileName] = metadataContent
        with open(self.outputJSON, "w") as outputFile:
            json.dump(specificJsonDictionary, outputFile, indent=4)

    def selectiveFileName(self):
        if os.path.isdir(self.filesToSearch):
            for csvFolderName in os.listdir(self.filesToSearch):
                with open(os.path.join(self.filesToSearch, csvFolderName)) as csvFile:
                    csvContent = csv.reader(csvFile)
                    for row in csvContent:
                        self.listOfFilesToSearch.append(row[0]) #change the method of accessign CSV depending on how your CSV is formatted
        elif os.path.isfile(self.filesToSearch) and self.is_csv(self.filesToSearch):
            try:
                with open(self.filesToSearch, "r") as csvFile:
                    csvContent = csv.reader(csvFile)
                    for row in csvContent:
                        self.listOfFilesToSearch.append(row[0]) #change the method of accessign CSV depending on how your CSV is formatted
            except csv.Error:
                print("Please only use CSV files or folder of CSV files")
                sys.exit(1)
            
if __name__ == "__main__":
    fullMetaJSON = "XMLProcessingAndTraining/ManuscriptMetadata/DocumentMetadata.json"
    specificMetaJSON = "XMLProcessingAndTraining/ManuscriptMetadata/SpecificMetadata.json"
    filesToSearch = "/Users/Jerry/Desktop/EEBOClassificationsCSV" #where you store the files you want to search for. The program is compatible for both CSV files or a folder of CSV files here.

    specificMetadataFinder = findSpecificMetadata(fullMetaJSON, specificMetaJSON, filesToSearch)
    specificMetadataFinder.selectiveFileName()
    specificMetadataFinder.findSpecificMeta()