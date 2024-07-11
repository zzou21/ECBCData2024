import nltk
filePath = "/Users/Jerry/Desktop/TXTFolder/A16864.txt"
keywords = ["powhatan", "virginia", "pocahontas", "america", "roanoke", "ratcliffe", "heathen", "chesapeake"]
'''raleigh 
colony 
indian 
croatoan 
savage 
massawomeck 
foreign 
sassafras 
inhabitant 
turkey 
unconvert 
pagan 
patawomeck 
werowance 
canoe 
chickahominy 
werowocomoco 
Ocanahowan
 '''

def processMainContent(contentPath):
    with open(contentPath, "r") as file:
        contentText = file.read()
    contentText = contentText.replace("\n", " ")
    tokenizedSentences = nltk.tokenize.sent_tokenize(contentText)
    tokenizedSentences = [sentence.strip() for sentence in tokenizedSentences if len(sentence.strip()) >= 30] # clear sentences that are too short to the point that it was mistakening tokenized or the tokenizer caught onto something uncessary.
    return tokenizedSentences

sentences = processMainContent(filePath)
for sentence in sentences:
    for word in keywords:
        if word in sentence:
            print(word, sentence + "\n")
