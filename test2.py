import json
# path = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/FindCosineSimilarityWords/decodeCosSimOutput/RollingUpdatesOutputs.json"
# existingJSON = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/FindCosineSimilarityWords/combined2023csvAndTFIDFFileNames.json"
# updatedJSON = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/updatedFilesToProcess.json"
# with open(path, "r") as jsonFile:
#     content = json.load(jsonFile)


# with open(existingJSON, "r") as existingFile:
#     existingContent = json.load(existingFile)

# dict = {}
# for name, data in existingContent.items():
#     if name not in content:
#         dict[name] = 1

# with open(updatedJSON, "w") as updateFile:
#     json.dump(dict, updateFile, indent=4)

dicti = {}
with open("/Users/Jerry/Desktop/600.txt", "r") as txtfile:
    content = txtfile.readlines()
    for file in content:
        dicti[file[:-5]] = 1
with open("/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/FindCosineSimilarityWords/600Files.json", "w") as dumpfile:
    json.dump(dicti, dumpfile, indent=4)