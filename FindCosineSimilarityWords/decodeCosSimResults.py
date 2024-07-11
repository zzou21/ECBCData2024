'''This auxiliary file turns cosine similarity outputted JSON into readable methods through customizable manipulation setings.'''

import json

def decode(JSONPath):
    decodedDictionary = {}
    with open(JSONPath, "r") as file:
        content = json.load(file)
    for fileName, basewordDict in content.items():
        decodedDictionary[fileName] = {}
        for baseword, keywordMetadataDict in basewordDict.items():
            if baseword not in decodedDictionary[fileName]:
                decodedDictionary[fileName][baseword] = []
            for keyword, metadata in keywordMetadataDict.items():
                if len(metadata) > 1:
                    decodedDictionary[fileName][baseword].append([(keyword, score) for score, sentence, index in metadata])
                    decodedDictionary[fileName][baseword]
                elif len(metadata) == 1:
                    decodedDictionary[fileName][baseword].append((keyword, metadata[0][0]))
    for fileName, content in decodedDictionary.items():
        print(f"For file: {fileName}")
        for baseword, score in content.items():
            print(f"For baseword {baseword}:")
            for tuple in score:
                print(tuple)

if __name__ == "__main__":
    codedJSONPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/FindCosineSimilarityWords/tesetOptimizer.json"
    decode(codedJSONPath)