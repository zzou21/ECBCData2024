'''This file is the main body of the multi-file Bible-non-Bible verse comparison machine program. It extracts potential Biblical references/citations from EEBO's XML files and store them in a JSON. Then, it calls on an external Python class object that drives the fine-tuned MacBERTh comparison machine to compare which verse was in the Geneva Bible and store the results in a separate JSON file.

Files to track:
BibleVerseDetectionTrainingImplementationPythonClassObject.py
Two JSON files (unfiltered and filtered Biblical references/citations)

Track "tunedModelPath," as it contains the path reference to the fine-tuned model.

This file also contains two more auxiliary functions, one for data wrangling and exploration while the other is for parsing only one XML file in case of debugging, experimentation, or emergency needs.

Author: Jerry Zou
'''
import xml.etree.ElementTree as ET
import os, random, json, csv, sys, numpy as np, matplotlib.pyplot as plt
# import pandas as pd, seaborn as sns

#the below import imports the external Python class object that drives the comparison model that compares whether a phrase is or is not in the Geneva Bible.
from BibleVerseDetectionTrainingImplementationPythonClassObject import BibleVerseComparison
tunedModelPath = "/Users/Jerry/Desktop/100Paramtest" #put the pathname of the fine-tuned model here that will be used by the BibleVerseComparison class object. Change this path name accordingly


'''Main Python class object:
This code cell is designed to search folders of XML files for general parsing purposes. The result will be stored in a JSON file. If running on local machine, make sure to add an iteration code so that the data gets pushed to JSON after a set amount of iterations so that the program doesn't exceed the local machine's memory storage space.

Currently, this program processes each file according to the line number of the italicized words to determine distribution.

We need a function to determine the total line number of every file path so that when we create dispersion plots, they are proportional to each file.

For the JSON file in which the output data is stored, the format for the dictionary is:
Original:
{filename : [{tag name: {line number : [sentence content, sentence content 2], line number : [sentence content, sentence content 2]}, tag name: {line number : [sentence content, sentence content 2], line number : [sentence content, sentence content 2]}}]}'''

class parseXMLFolder:
    def __init__(self, jsonPathRaw, jsonPathFiltered, folderPathEEBOOne, folderPathEEBOTwo, tag, csvFolder):
        self.jsonPathRaw = jsonPathRaw
        self.folderPaths = [folderPathEEBOOne, folderPathEEBOTwo] #lsit of two lists of file paths
        self.tagsAsJson = tag
        self.tagAsList = []
        self.csvFolder = csvFolder
        self.selectedFileNames = []

    def turnTagsIntoList(self):
        with open(self.tagsAsJson, "r") as file:
            tagsJsonFormat = json.load(file)
        for tagSubList in tagsJsonFormat.values():
            for specificTag in tagSubList:
                self.tagAsList.append(specificTag)
        print(type(self.tagAsList))

    def openFiles(self, filePathsList):
        contentDictionary = {}
        for filePath in filePathsList:
            with open(filePath, "r", encoding="utf-8") as file:
                fileContent = file.read()  # Read the entire file content
            fileDictionaryKey = os.path.basename(filePath)[:-4]
            contentDictionary[fileDictionaryKey] = [{}]  # Initialize with a list containing an empty dictionary
            tree = ET.ElementTree(ET.fromstring(fileContent))
            root = tree.getroot()

            def extract_text(element):
                text = element.text or ""
                for subelement in element:
                    text += ET.tostring(subelement, encoding='unicode', method='text')
                    if subelement.tail:
                        text += subelement.tail
                return text
            
            for oneTag in self.tagAsList:
                tagDict = {}
                for tag in root.iter(oneTag):
                    if oneTag == "Q" and tag.find("BIBL") is not None:
                        tag_text = extract_text(tag)
                        tag_str = ET.tostring(tag, encoding='unicode').strip()
                        start_index = fileContent.find(tag_str)
                        if start_index == -1:
                            continue
                        lineNum = fileContent[:start_index].count('\n') + 1
                        noNewLineTagText = tag_text.replace("\n", " ")
                        if lineNum not in tagDict:
                            tagDict[lineNum] = []
                        tagDict[lineNum].append(noNewLineTagText)
                    else:
                        tag_text = extract_text(tag)
                        tag_str = ET.tostring(tag, encoding='unicode').strip()
                        start_index = fileContent.find(tag_str)
                        if start_index == -1:
                            continue
                        lineNum = fileContent[:start_index].count('\n') + 1
                        noNewLineTagText = tag_text.replace("\n", " ")
                        if lineNum not in tagDict:
                            tagDict[lineNum] = []
                        tagDict[lineNum].append(noNewLineTagText)
                contentDictionary[fileDictionaryKey][0][oneTag] = tagDict
        #print(contentDictionary)
        return contentDictionary

    def addBatchToJson(self, newBatch):
        try:
            with open(self.jsonPathRaw, "r", encoding="utf-8") as file:
                currentData = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            currentData = {}  # Added exception handling for file not found or invalid JSON
        currentData.update(newBatch)
        with open(self.jsonPathRaw, "w", encoding="utf-8") as file:
            json.dump(currentData, file, indent=4)

    def preprocessForJson(self):
        allFileNames = []
        for folder in self.folderPaths:
            for root, subFolder, files in os.walk(folder):
                for fileName in files:
                    if fileName.endswith(".xml"): # This ensures that only XML files are processed
                        if self.selectedFileNames: #If there are designated files to read (aka if the self.selectedFileNames list is not empty)
                            for selectedXMLs in self.selectedFileNames:
                                if fileName.contains(selectedXMLs): # only reads designated files
                                    fullFilePathName = os.path.join(root, fileName)
                                    allFileNames.append(fullFilePathName)
                        else: # read all XML files in the folder
                            fullFilePathName = os.path.join(root, fileName)
                            allFileNames.append(fullFilePathName)
        batchLength = 1 # The "batchLength" variable is designed to be a safety pin. It limits how many files are being processed at once. Change this number depending on how many files the device memory allows to store. If you are using a large machine, such as Duke Cluster Computers, this number shoudn't matter as much. Just don't crash whichever device you are using.
        for fileCount in range(0, len(allFileNames), batchLength):
            batchXML = allFileNames[fileCount:fileCount + batchLength]
            newParsing = self.openFiles(batchXML)
            self.addBatchToJson(newParsing)
    
    #This function processes uncleaned Biblical (italics) extracts). Specific function: regulate workflow according to how many key:value pairs could be processed at the same time so to not overburden device memory
    def filterParsedQuotes(self, jsonPathRaw, jsonPathFiltered):
        def processBatch(batch):
            batchDictionary = {}
            comparisonMachine = BibleVerseComparison(tunedModelPath) #initiates the Python object that drives the comparison model. The function name within the object is "checkBibleVerse". When implementing this model further down in this function, use comparisonMachine.checkBibleVerse(...). Put the sentence to check, or a reference to that sentence, in the parentheses.
            for pair in batch:
                fileName = pair[0] #grabs the file name, e.g. "A00001"
                tagsDictionary = pair[1][0] #grabs the dictionary containing all tags
                batchDictionary[fileName] = [{}]
                for tagName, lineDictionary in tagsDictionary.items(): #iterate through tags dictionary
                    newTag = {} #starts a new dictionary at the tag level
                    for lineNumber, sentences in lineDictionary.items():
                        newSentences = [] 
                        for phrase in sentences: #base level. Access phrase content in this for loop.
                            if len(phrase) >= 20: #ASSUMPTION: Biblical references will be longer than 20 characters
                                if comparisonMachine.checkBibleVerse(phrase) is True:
                                    # FOR DEBUG if "Sea" in phrase:
                                    # editedPhrase = phrase.lower() #EXAMPLE OPERATION replace this with needed changes
                                    newSentences.append(phrase) #put the changed content back to the list
                    # if newSentences: # this If statement will control that only if, after editing, a line still with content will be added back to the new dictionary.
                        newTag[lineNumber] = newSentences
                    # if newTag : #this If statement ensures that only, after editing, a tag dictionary with content will be added back to the new dictionary.
                    batchDictionary[fileName][0][tagName] = newTag
            return batchDictionary
        
        def addFilteredDictionarytoJson(newBatch):
            try:
                with open(jsonPathFiltered, "r", encoding="utf-8") as file:
                    currentData = json.load(file)
            except (FileNotFoundError, json.JSONDecodeError):
                currentData = {}
            currentData.update(newBatch)
            with open(jsonPathFiltered, "w", encoding="utf-8") as file:
                json.dump(currentData, file, indent=4)
            
        # newDictionary = {} #starts a new dictionary to store edited content to.
        with open(jsonPathRaw, "r") as rawQuotesFile:
            rawQuotesContent = json.load(rawQuotesFile)
        listContent = list(rawQuotesContent.items()) #JSON dictionary to list of tuples

        batchSizeFiltering = 2
        for dictionaryCount in range(0, len(listContent), batchSizeFiltering): #workflow control so the processing doesn't take up too much device memory. Change batchSizeFiltering value to how much you think your device memory could hangle in each computation. The batchSizeFiltering refers to the number of filename dictionaries in the raw XML extract JSON file.
            batch = listContent[dictionaryCount:dictionaryCount+batchSizeFiltering]
            batchDictionary = processBatch(batch) # call filtering dictionary
            addFilteredDictionarytoJson(batchDictionary)

    #This function prepares which files the system will parse
    def selectiveFileName(self):
        if os.path.isdir(self.csvFolder):
            for csvFolderName in os.listdir(self.csvFolder):
                with open(os.path.join(self.csvFolder, csvFolderName)) as csvFile:
                    csvContent = csv.reader(csvFile)
                    for row in csvContent:
                        self.selectedFileNames.append(row[0]) #change the method of accessign CSV depending on how your CSV is formatted
        elif os.path.isfile(self.csvFolder) and self.is_csv(self.csvFolder):
            try:
                with open(self.csvFolder, "r") as csvFile:
                    csvContent = csv.reader(csvFile)
                    for row in csvContent:
                        self.selectedFileNames.append(row[0]) #change the method of accessign CSV depending on how your CSV is formatted
            except csv.Error:
                print("Please only use CSV files or folder of CSV files")
                sys.exit(1)
        print(self.selectedFileNames)

'''Auxiliary function #1: INFORMATIONAL DATA WRANGLING CODE. CUSTOMIZE OUTPUT TO FIT THE GOAL OF EACH USAGE. These two functions grab the file names of all XML files in both Phases One and Two. This code cell is more for informational purposes, such as checking the number of files in each directory, etc.'''
class AuxiliaryDataExploration:
    def __init__(self, folderPathOne, folderPathTwo):
        self.folderPathOne = folderPathOne
        self.folderPathTwo = folderPathTwo
        self.fileNameListOne = []
        self.fileNameListTwo = []
    def findAllFileNamesOne(self):
        for root, subFolderList, files in os.walk(self.folderPathOne):
            for fileName in files:
                if fileName.endswith(".xml"):
                    fullFilePathName = os.path.join(root, fileName)
                    self.fileNameListOne.append(os.path.basename(fullFilePathName)[:-4])
    def findAllFileNamesTwo(self):
        for root, subFolderList, files in os.walk(self.folderPathTwo):
            for fileName in files:
                if fileName.endswith(".xml"):
                    fullFilePathName = os.path.join(root, fileName)
                    self.fileNameListTwo.append(os.path.basename(fullFilePathName)[:-4])

'''Auxiliary function #2: This class object parces one single XML'''
class AuxiliarySingleXMLParcing:
    def __init__(self, xmlPath):
        self.xmlPath = xmlPath
        self.lineNumberAndContentSingleFile = []
    def singleFileSearch(self):
        with open(self.xmlPath, "r") as file:
            content = file.readlines()
        for lineNumber, lineContent in enumerate(content, start=1):
            try:
                root = ET.fromstring(lineContent)
                HITags = root.findall("HI")  # change the tag you are looking for here
                for tag in HITags:
                    self.lineNumberAndContentSingleFile.append((lineNumber, tag.text))
            except ET.ParseError:
                continue

if __name__ == "__main__":
    '''--- Main Bible-non-Bible Comparison model ---'''
    jPathRawExtracts = "StoringItalicsAndLineNumber.json" #change pathname accordingly
    jpathFiltered = "XMLProcessingAndTraining/filteredXMLBibleRef.json" #change pathname accordingly
    EEBOOne = "/Users/Jerry/Desktop/TestFolder" #change pathname accordingly
    EEBOTwo = "/Users/Jerry/Desktop/A01" #change pathname accordingly
    tagPath = "XMLCitationTags.json" #change pathname accordingly
    docClustersFolder = "/Users/Jerry/Desktop/EEBOClassificationsCSV" #please only use a folder of CSV files or a single CSV file.

    parcer = parseXMLFolder(jPathRawExtracts, jpathFiltered, EEBOOne, EEBOTwo, tagPath, docClustersFolder)
    parcer.selectiveFileName() # Call this function if we need to parce a selected group of files
    # parcer.turnTagsIntoList()
    # parcer.preprocessForJson()
    # parcer.filterParsedQuotes(jPathRawExtracts, jpathFiltered)


    ''' --- Using auxiliary function one to explore data -- '''
    #make sure to change the class object according to your need
    auxFolderPathOne = "/Volumes/JZ/EEBOData+2024/eebo_phase1/P4_XML_TCP" #change pathname accordingly
    auxFolderPathTwo = "/Volumes/JZ/EEBOData+2024/eebo_phase2/P4_XML_TCP_Ph2" #change pathname accordingly
    # Uncomment function calls and print statements below and add customized code below when needed. These four lines included are default class object calls:
    
    # auxiliaryOne = AuxiliaryDataExploration(auxFolderPathOne, auxFolderPathTwo) #object call
    # auxiliaryOne.findAllFileNamesOne()
    # auxiliaryOne.findAllFileNamesTwo()
    # print(auxiliaryOne.fileNameListOne)
    # print(auxiliaryOne.fileNameListTwo)

    ''' --- Using auxiliary function two as single XML call for testing, debugging, emergency, experimentation, and other purposes -- '''
    xmlSingleFilePath = "/Users/Jerry/Desktop/A0/A00002.P4.xml" #change pathname accordingly
    # Uncomment function calls and print statements below and add customized code below when needed. These four lines included are default class object calls:

    # auxiliaryTwo = AuxiliarySingleXMLParcing(xmlSingleFilePath)
    # auxiliaryTwo.singleFileSearch()
    # print(auxiliaryTwo.lineNumberAndContentSingleFile)