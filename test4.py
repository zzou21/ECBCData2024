import json, os

folder = "/Users/Jerry/Desktop/Virginia_Clean_Punc"
existingDocuments = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/FindCosineSimilarityWords/80VAFiles.json"
jsonLoc = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/200FilesVirginia1606-1625.json"

with open(existingDocuments, "r") as file:
    existingFileNames = json.load(file)


counter = 0
dictT = {}
for fileName in os.listdir(folder):
    if fileName.endswith(".txt"):
        if fileName[:-4] not in existingFileNames:
            counter += 1
            dictT[fileName[:-4]] = 1


with open(jsonLoc, "w") as file:
    json.dump(dictT, file, indent=4)