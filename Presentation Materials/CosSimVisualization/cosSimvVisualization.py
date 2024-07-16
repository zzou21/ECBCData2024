'''This program tests out generating visualizations for cosine similarity outputs.

Author: Jerry Zou'''
import json

class visualizeCosSimOutputs:
    def __init__(self, jsonPath):
        self.jsonPath = jsonPath
        self.jsonPathContentDictionary = None
        with open(jsonPath, "r") as jsonFile:
            self.jsonPathContentDictionary = json.load(jsonFile)
        
    def heatMap(self):
        dataVis = {}
        dataVis["Output Words"] = []
        for fileName, basewordDict in self.jsonPathContentDictionary.items():
            for baseword, keywordList in basewordDict.items():
                if baseword not in dataVis: dataVis[baseword] = []
                print(baseword)
                # for keywordInfo in keywordList:

    def networkRelations(self):
        appearanceDictionary = {}
        for filename, basewordDict in self.jsonPathContentDictionary.items():
            for baseword, keywordList in basewordDict.items():
                # print(type(keywordList))
                for keywordSelf, keywordInfo in keywordList.items():
                    cosSimScore = keywordInfo[0][0]
                    contextSentence = keywordInfo[0][1]
                    # print(keywordInfo[0][1])
                    if keywordSelf not in appearanceDictionary:
                        appearanceDictionary[keywordSelf] = {}
                    if baseword not in appearanceDictionary[keywordSelf]:
                        appearanceDictionary[keywordSelf][baseword] = []
                    appearanceDictionary[keywordSelf][baseword].append(filename)
                    # appearanceDictionary[keywordSelf][baseword].append(contextSentence)

        counter = 0
        print(appearanceDictionary)
        for returnedWord, appearance in appearanceDictionary.items():
            # if len(appearance) >= 5:
            # print()
            counter += 1
        print(appearanceDictionary)

if __name__ == "__main__":
    jsonPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/FindCosineSimilarityWords/eightSermonsOutputs.json"
    visCosSim = visualizeCosSimOutputs(jsonPath)
    visCosSim.networkRelations()