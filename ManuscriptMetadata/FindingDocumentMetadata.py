'''This program pairs the numeric file name (such as "A00001") with the actual title, author, publication year, and other metadata of the manuscript. The data is then stored in a JSON document.
Currently, this program considers the file numerical name, manuscript title, author, publication year, publisher, and publication city.

Author: Jerry Zou'''

import os, json, re, xml.etree.ElementTree as ET

class metaDataIdentifier:
    def __init__(self, folderPath1, folderPath2, mainLevelKeyList, XMLTagList, modernWordsToFilter, exportJSONPath):
        self.folderspath = [folderPath1, folderPath2]
        self.mainLevelKeyList = mainLevelKeyList
        self.XMLTagList = XMLTagList
        self.modernWordsToFilter = modernWordsToFilter
        self.exportJSONPath = exportJSONPath
        self.dictionary = {key: {} for key in mainLevelKeyList}

    # def findingMetadata(self):
    #     for mainSubFolder in self.folderspath:
    #         for root, secondSubFolder, fileNames in os.walk(mainSubFolder):
    #             # print(f"root: {root}")
    #             # print(f"second Sub Folder: {secondSubFolder}")
    #             for key in self.mainLevelKeyList:
    #                 if key in root:
    #                     self.dictionary[key] = {}
    #             for files in fileNames:
    #                 if files.endswith(".xml") and not files.startswith("._"):
    #                     if "phase1" in os.path.join(root, files):
    #                         self.dictionary["phase1"][files[:-7]] = self.metadataExtract(os.path.join(root, files), self.XMLTagList, self.modernWordsToFilter)
    #                         print(f"phase1 file: {files[:-7]}")
    #                     if "phase2" in os.path.join(root, files):
    #                         self.dictionary["phase2"][files[:-7]] = self.metadataExtract(os.path.join(root, files), self.XMLTagList, self.modernWordsToFilter)
    #                         print(f"phase2 file: {files[:-7]}")

    #     return self.dictionary
    
    def findingMetadata(self):
        for mainSubFolder in self.folderspath:
            for root, _, fileNames in os.walk(mainSubFolder):
                for key in self.mainLevelKeyList:
                    if key in root:
                        current_key = key
                        break
                else:
                    continue

                for files in fileNames:
                    if files.endswith(".xml") and not files.startswith("._"):
                        file_path = os.path.join(root, files)
                        content = self.metadataExtract(file_path, self.XMLTagList, self.modernWordsToFilter)
                        self.dictionary[current_key][files[:-4]] = content
                        # print(f"{current_key} file: {files[:-4]}")
        return self.dictionary

    def metadataExtract(self, filePath, tagsList, filterModernWords):
        contentDictionary = {}
        for tag in tagsList:
            contentDictionary[tag] = []
        with open(filePath, "r", encoding="utf-8", errors="replace") as file:
            fileContent = file.read().replace("�", ".").strip()  # Read the entire file content and replace non utf-8 compatible characters with a period.
        fileContent = re.sub(r'^<xml ', '<?xml ', fileContent) #in case the XML file has a typo with the leading <?xml> tag. This is specific for EEBO since one EEBO XML file, B00838, forgot to add question mark.
        fileContent = re.sub(r'<!DOCTYPE [^>]*>', '', fileContent) 
        if not fileContent:
            print(f"Skipping empty file: {filePath}")
        try:
            tree = ET.ElementTree(ET.fromstring(fileContent))
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"Error parsing {filePath}: {e}")
            print(f"Problematic content: {fileContent.splitlines()[2]}")  
        def extract_text(element):
            text = element.text or ""
            for subelement in element:
                text += ET.tostring(subelement, encoding='unicode', method='text')
                if subelement.tail:
                    text += subelement.tail
            return text
        for oneTag in tagsList:
            for tag in root.iter(oneTag):
                tagText = extract_text(tag)
                if tagText not in contentDictionary[oneTag]:
                    filterEEBOModernText = self.filterMetadata(filterModernWords, tagText)
                    if filterEEBOModernText: contentDictionary[oneTag].append(tagText)
        return contentDictionary

    def filterMetadata(self, filterList, inputString):
        # print(filterList)
        modernDatePattern = re.compile(r'\b\d{4}-\d{2}\b')
        if modernDatePattern.search(inputString):
            return False
        for phrase in filterList:
            if phrase in inputString:
                return False
        return True
    
    def exportJSON(self):
        with open(self.exportJSONPath, "w") as fileJSON:
            json.dump(self.dictionary, fileJSON, indent=4)

if __name__ == "__main__":
    folder1 = "/Volumes/JZ/EEBOData+2024/eebo_phase1/P4_XML_TCP"
    folder2 = "/Volumes/JZ/EEBOData+2024/eebo_phase2/P4_XML_TCP_Ph2"
    # folder1 = "/Users/Jerry/Desktop/XMLTestFolder1"
    # folder2 = "/Users/Jerry/Desktop/XMLTestFolder2"
    mainLevelKeyList = ["phase1", "phase2"]
    # mainLevelKeyList = ["Folder1", "Folder2"]
    modernWordsToFilter = ["Ann Arbor", "Text Creation Partnership", "EEBO-TCP Phase 1", "EEBO-TCP Phase 2", "EEBO-TCP", "Early English books online."]
    XMLTagList = ["TITLE", "AUTHOR", "PUBPLACE", "PUBLISHER", "DATE"]
    exportJSON = "ManuscriptMetadata/DocumentMetadata.json"

    filteringMeta = metaDataIdentifier(folder1, folder2, mainLevelKeyList, XMLTagList, modernWordsToFilter, exportJSON)
    filteringMeta.findingMetadata()
    filteringMeta.exportJSON()