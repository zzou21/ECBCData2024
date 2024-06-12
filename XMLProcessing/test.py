import os
import json
import xml.etree.ElementTree as ET

filePath = "/Users/Jerry/Desktop/A00008.P4.xml"

contentDictionary = {}
tagAsList = ["HI", "HEAD"]  # List of tags you want to check

with open(filePath, "r", encoding="utf-8") as file:
    fileContent = file.read()  # Read the entire file content
fileDictionaryKey = os.path.basename(filePath)[:-4]
contentDictionary[fileDictionaryKey] = [{}]  # Initialize with a list containing an empty dictionary

# Parse the XML content
tree = ET.ElementTree(ET.fromstring(fileContent))
root = tree.getroot()

# Function to recursively extract all text from a tag including nested tags
def extract_text(element):
    text = element.text or ""
    for subelement in element:
        text += ET.tostring(subelement, encoding='unicode', method='text')
        if subelement.tail:
            text += subelement.tail
    return text

# Traverse and find all specified tags
for oneTag in tagAsList:
    tagDict = {}  # Dictionary to hold line numbers and texts for the current tag
    for tag in root.iter(oneTag):
        tag_text = extract_text(tag)
        tag_str = ET.tostring(tag, encoding='unicode').strip()
        start_index = fileContent.find(tag_str)
        if start_index == -1:
            continue
        lineNum = fileContent[:start_index].count('\n') + 1
        if lineNum not in tagDict:
            tagDict[lineNum] = []
        tagDict[lineNum].append(tag_text)
    contentDictionary[fileDictionaryKey][0][oneTag] = tagDict  # Add the tagDict to the main dictionary

print(contentDictionary)

'''
from bs4 import BeautifulSoup

filePath = "/Users/Jerry/Desktop/A0/A00011.P4.xml"

contentDictionary = {}
tagAsList = ["HI"]

with open(filePath, "r", encoding="utf-8") as file:
    fileContent = file.read()  # Read the entire file content
fileDictionaryKey = os.path.basename(filePath)[:-4]
print("Dict key:", fileDictionaryKey)
contentDictionary[fileDictionaryKey] = {}
print("content dictionary:", contentDictionary)

soupContent = BeautifulSoup(fileContent, "xml")  # Parse the entire file content

def get_line_number(tag, content):
    start_index = content.find(str(tag))
    return content[:start_index].count('\n') + 1

for oneTag in tagAsList:
    taggedContent = soupContent.find_all(oneTag)  # Find all tags in the list
    for tag in taggedContent:
        tag_str = str(tag)
        start_index = fileContent.find(tag_str)
        lineNum = get_line_number(tag, fileContent)
        if lineNum not in contentDictionary[fileDictionaryKey]:
            contentDictionary[fileDictionaryKey][lineNum] = []
        contentDictionary[fileDictionaryKey][lineNum].append(tag.text)
        # Remove the processed tag from the content to avoid duplicate counting
        fileContent = fileContent.replace(tag_str, ' ' * len(tag_str), 1)

print(contentDictionary)
'''