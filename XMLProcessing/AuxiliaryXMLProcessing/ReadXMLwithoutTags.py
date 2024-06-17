'''For whenever this is necessary, this code reads XML files without its embedded tags. This allows users to turn EEBO's marked-up XML's into raw plain text txts.

Author: Jerry Zou'''
import xml.etree.ElementTree as ET
import json, os, nltk, csv
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

#This function removes all tags from an XML file and returns just the plain text
def removeTags(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    def extract_text(element):
        textContent = ''
        if element.text:
            textContent += element.text
        for child in element:
            textContent += extract_text(child)
            if child.tail:
                textContent += child.tail
        return textContent
    textContent = extract_text(root)
    return textContent

#This function cleans the content returned by the fucntion "removeTags" by removing new lines and unecessayr spacing
def cleaning(text):
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = ' '.join(text.split())
    return text

# This function removes references already mentioned in the JSON file from the paragraphs returned by the function "cleaning". This is to prepare the documents for training so that the JSON file contains the important markers from these documents that the users have labeled while the output of this function will only include the parts of the corpus that do not include the designated content listed in the JSON file. Make sure the user updates the JSON file before running this function.
def deleteReferences(jsonPath, text, dictionaryKey):
    with open(jsonPath, "r") as jsonFile:
        jsonContent = json.load(jsonFile)
    if dictionaryKey in jsonContent:
        value = jsonContent[dictionaryKey]
    else:
        print(f"'{dictionaryKey}' not found in JSON")
        return text
    
    for tag, value in value[0].items():
        for lineNumber, content in value.items():
            for phrase in content:
                if len(phrase) > 35:
                    text = text.replace(phrase, "")

    # Here, the function uses NLTK Punkt tokenizer to conduct sentence segmentation on the text so that it could be returned as list of sentences. Change the four lines of below according to the specific need of the context of usage. Feel free to comment out the four lines below if there is no more processing needed after removing the designated phrases in JSON or edit the four lines below to suit your need.
    punkt_param = PunktParameters()
    sentence_tokenizer = PunktSentenceTokenizer(punkt_param)
    tokenizedSentences = sentence_tokenizer.tokenize(text)
    returnListOfSentences = [sentences for sentences in tokenizedSentences if len(sentences) > 20]
    # FOR DEBUG print(type(sentences))
    # FOR DEBUG for i, sentence in enumerate(sentences):
    # FOR DEBUG    print(f"Sentence {i+1}: {sentence}")
    #FOR DEBUG print(f"length: {dictionaryKey} {len(returnListOfSentences)}")
    return returnListOfSentences

# This is the main function that calls all functions above and combines the output of the functions above into one data variable (let that be a list or dictionary, depending on the specific usage that the user can edit themselves)
def processFolder(folderPath, jsonPath):
    dictionaryKeyList = []
    allNonBiblicalSentencesMainList = []
    numericallyLabeledInDictionary = {}

    #Traverse a folder that holds files.
    for filename in os.listdir(folderPath):
        if filename.endswith(".xml"):
            file_path = os.path.join(folderPath, filename)
            dictionaryKey = filename[:-4]
            dictionaryKeyList.append(dictionaryKey)
            textContent = removeTags(file_path)
            cleanedContent = cleaning(textContent)
            content = deleteReferences(jsonPath, cleanedContent, dictionaryKey)
            for sentence in content: #combine all outputs into one sentence.
                allNonBiblicalSentencesMainList.append(sentence)
    # FOR DEBUG print(f"length: {len(allNonBiblicalSentencesMainList)}")
    # FOR DEBUG with open("XMLProcessing/AuxiliaryXMLProcessing/test.txt", "w") as file:
    # FOR DEBUG    for items in allNonBiblicalSentencesMainList:
    # FOR DEBUG        file.write(items + "\n")
    # FOR DEBUG for i, sentence in enumerate (allNonBiblicalSentencesMainList):
    # FOR DEBUG    print(f"sentence {i+1} {sentence}")
    for i, sentence in enumerate(allNonBiblicalSentencesMainList):
        numericallyLabeledInDictionary[i+1] = sentence
    print(len(numericallyLabeledInDictionary))
    return numericallyLabeledInDictionary

if __name__ == "__main__":
    JSONPath = "XMLProcessing/StoringItalicsAndLineNumber.json"
    folderPath = "/Users/Jerry/Desktop/TestFolder" 
    finalDictionary = processFolder(folderPath, JSONPath)

    '''Uncomment the code below when the user is ready to load content into a CSV'''
    # output_csv_path = "output.csv"
    # with open(output_csv_path, 'w', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(["ID", "Sentence"])
        
    #     for key, value in finalDictionary.items():
    #         csvwriter.writerow([key, value])
    # print(f"Data has been written to {output_csv_path}")