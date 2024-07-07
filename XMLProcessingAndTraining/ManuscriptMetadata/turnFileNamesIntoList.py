import json
from FindSpecificMetadata import findSpecificMetadata

def txtNamesToList(txtFile):
    fullMetaJSON = "XMLProcessingAndTraining/ManuscriptMetadata/DocumentMetadata.json"
    specificMetaJSON = "XMLProcessingAndTraining/ManuscriptMetadata/SpecificMetadata.json"
    filesToSearch = "/Users/Jerry/Desktop/EEBOClassificationsCSV" #where you store the files you want to search for. The program is compatible for both CSV files or a folder of CSV files here.

    with open(txtFile, "r") as txtF: content = txtF.read()
    content = content.replace("\n", ", ")
    content = content.replace(", ", " ")
    nameList = list(set(name.strip() for name in content.split()))
    fileWithNames = findSpecificMetadata(fullMetaJSON, specificMetaJSON, filesToSearch)
    fileWithNames.selectiveFileName()
    csvNameList = fileWithNames.listOfFilesToSearch
    combinedList = list(set(nameList+csvNameList))
    return combinedList

if __name__ == "__main__":
    txtFile = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/XMLProcessingAndTraining/ManuscriptMetadata/2023TFIDFDocs.txt"
    jsonFilePath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/XMLProcessingAndTraining/ManuscriptMetadata/combined2023csvAndTFIDFFileNames.json"
    listNames = {v: k+1 for k, v in enumerate(txtNamesToList(txtFile))}
    with open(jsonFilePath, "w") as file: json.dump(listNames, file, indent=4)