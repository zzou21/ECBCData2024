'''For whenever this is necessary, this code reads XML files without its embedded tags. This allows users to turn EEBO's marked-up XML's into raw plain text txts.

Author: Jerry Zou'''

import xml.etree.ElementTree as ET
import json, os
# from transformers import AutoTokenizer, AutoModelForTokenClassification
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

def removeTags(folderPath):

    tree = ET.parse( )
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

def cleaning(text):
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = ' '.join(text.split())
    return text

def deleteReferences(jsonPath, text, dictionaryKey):
    with open(jsonPath, "r") as jsonFile:
        jsonContent = json.load(jsonFile)
    if dictionaryKey in jsonContent:
        value = jsonContent[dictionaryKey]
    else:
        print(f"'{dictionaryKey}' not found in JSON")
    for tag, value in value[0].items():
        for lineNumber, content in value.items():
            for phrase in content:
                if len(phrase) > 35:
                    text = text.replace(phrase, "")
                
    textDictionary = {}
    punkt_param = PunktParameters()
    sentence_tokenizer = PunktSentenceTokenizer(punkt_param)
    sentences = sentence_tokenizer.tokenize(text)
    for i, sentence in enumerate(sentences):
        print(f"Sentence {i+1}: {sentence}")

    return text

def processFolderGeneral(folderPath, jsonPath):
    for filename in os.listdir(folderPath):
        if filename.endswith(".xml"):
            file_path = os.path.join(folderPath, filename)
            dictionaryKey = filename[:-4]
            print(file_path)
            print(dictionaryKey)

JSONPath = "XMLProcessing/StoringItalicsAndLineNumber.json"
XMLFolderPath = "/Users/Jerry/Desktop/BibleTrainingEEBOXML"
dictionaryKey = "A00002.P4"
# textContent = removeTags(XMLFolderPath)
# cleanedContent = cleaning(textContent)
# deleteReferences(JSONPath, cleanedContent, dictionaryKey)
processFolderGeneral(XMLFolderPath, JSONPath)