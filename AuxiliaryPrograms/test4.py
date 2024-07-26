import json, os

folder = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/2023_cleaned"
existingDocuments = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/FindCosineSimilarityWords/80VAFiles.json"
jsonLoc = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/FindCosineSimilarityWords/80VAFiles.json"

with open(existingDocuments, "r") as file:
    existingFileNames = json.load(file)


counter = 0
dictT = {}
for fileName in os.listdir(folder):
    if fileName.endswith(".txt"):
        counter += 1
        dictT[fileName[:-4]] = 1


with open(jsonLoc, "w") as file:
    json.dump(dictT, file, indent=4)