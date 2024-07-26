'''
This program searches for files between 1606-1625:

import json

metaJSON = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/XMLProcessingAndTraining/ManuscriptMetadata/cleanedDocumentMetadata.json"

second = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/FindCosineSimilarityWords/600Files.json"

outputJson = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/400FilesNames.json"

with open(metaJSON, "r") as file:
    content = json.load(file)

with open(second, "r") as file:
    contentsecond = json.load(file)

dictionary = {}

counter = 0
for phase, fileDict in content.items():
    for filename, metadataDict in fileDict.items():
        if filename in contentsecond:
            # print(metadataDict["DATE"])
            if metadataDict["DATE"]:
                if len(metadataDict["DATE"]) > 1:
                    if 1605 < min(metadataDict["DATE"]) < 1626:
                        dictionary[filename] = 1
                        counter += 1
                elif 1605 < metadataDict["DATE"][0] < 1626:
                    dictionary[filename] = 1
                    counter += 1

print(counter)

with open(outputJson, "w") as file:
    json.dump(dictionary, file, indent=4)
'''