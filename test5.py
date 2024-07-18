#This is to combine the batched cosine similarity jobs for DCC into one JSON

# dict1 = {1: 2, 3: 4, 5: 6}
# dict2 = {7:8, 9: 0}
# dict3 = {"3": 10, "5": 20}

# newDict = dict1 | dict2 | dict3
# print(newDict)
import json

batch1 = "/Users/Jerry/Desktop/batch1Storage.json"
batch2 = "/Users/Jerry/Desktop/batch2Storage.json"
batch3 = "/Users/Jerry/Desktop/batch3Storage.json"
batch4 = "/Users/Jerry/Desktop/batch4Storage.json"
sermons = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/FindCosineSimilarityWords/eightSermonsFileNames.json"
sermonsStorage = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/FindCosineSimilarityWords/eightSermonsOutputs.json"

with open(batch1, "r") as one:
    oneCont = json.load(one)
with open(batch2, "r") as two:
    twoCont = json.load(two)
with open(batch3, "r") as three:
    threeCont = json.load(three)
with open(batch4, "r") as four:
    fourCont = json.load(four)

with open(sermons, "r") as sermonsfile:
    sermonsNames = json.load(sermonsfile)

newDictionary = oneCont | twoCont | threeCont | fourCont

newSermonsDictionary = {key: values for key, values in newDictionary.items() if key in sermonsNames}

print(len(newSermonsDictionary))

with open(sermonsStorage, "w") as fileStorage:
    json.dump(newSermonsDictionary, fileStorage, indent=4)