'''This program assumes that the user has stored all the word embedding coordinates (vectors) into a JSON file for faster computation. This program, instead of embedding the content of the corpus each time the user runs analysis, will directly access the stored embedding.

Author: Jerry Zou'''

import numpy as np, json, torch, heapq
from transformers import AutoTokenizer, AutoModel

def accessJSON(JSONPath, notReturn):
    with open(JSONPath, "r") as jsonFile:
        content = json.load(jsonFile)

    newDictionary = {}
    for word, coordinate in content.items():
        if word not in notReturn:
            newDictionary[word] = np.array(coordinate)
    return content

def findKeywordEmbedding(keywordList, model, tokenizer):
    embeddingDictionary = {}
    for keyword in keywordList:
        inputs = tokenizer(keyword, return_tensors='pt')
        print(f"Embedding word: {keyword}")

        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        for i, token in enumerate(tokens):
            embeddingDictionary[token] = embeddings[0, i, :].numpy()
    print(embeddingDictionary)
    
    return embeddingDictionary

def computeTopSimilarwords(JSONPath, keywordList, model, tokenizer, numOfWords, notReturn):
    existingEmbedding = accessJSON(JSONPath, notReturn)
    keywordEmbedding = findKeywordEmbedding(keywordList, model, tokenizer)
    topSimilarWords = {}
    for keyword, keyEmbedding in keywordEmbedding.items():
        similarWords = []
        for word, embedding in existingEmbedding.items():
            if word in notReturn:
                pass
            else:
                cosine_similarity = np.dot(embedding, keyEmbedding) / (np.linalg.norm(embedding) * np.linalg.norm(keyEmbedding))
                if len(similarWords) < numOfWords:
                    heapq.heappush(similarWords, (cosine_similarity, word))
                else:
                    # If the current similarity is greater than the smallest similarity in the heap, replace it
                    heapq.heappushpop(similarWords, (cosine_similarity, word))
        topSimilarWords[keyword] = sorted(similarWords, key=lambda x: x[0], reverse=True)
    return topSimilarWords

# def findKeyword(keywordList, JSONPath):
#     with open(JSONPath, "r") as jsonFile:
#         content = json.load(jsonFile)
#     for word in JSONPath.keys():
#         if word.startswith("laz"):
#             print(word)


model_name = "emanjavacas/MacBERTh"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

keywordList = ["plantation", "london", "obedience", "treachery", "dutie", "lazy"]
JSONPath = "/Users/Jerry/Desktop/embeddingCoordinates.json"
numOfWords = 50
notReturn = ["iiii", "xvi", "vii", "xiv", "viii", "xv", "xv", "ii", "ix", "xii", "xvii", "xxii", "ʒ", "✚", "finis", "init", "☊", "§", "ibidem"]

result = computeTopSimilarwords(JSONPath, keywordList, model, tokenizer, numOfWords, notReturn)
for keyword, tuplelist in result.items():
    
    print(f"For baseword: {keyword}")
    for tuple in tuplelist:
        print((tuple[1], tuple[0]))
