import json, os

folder = "/Users/Jerry/Desktop/Team2024cleaned2023"
jsonLoc = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/updatedFilesToProcess.json"

dictT = {}
for fileName in os.listdir(folder):
    if fileName.endswith(".txt"):
        dictT[fileName[:-4]] = 1

with open(jsonLoc, "w") as file:
    json.dump(dictT, file, indent=4)