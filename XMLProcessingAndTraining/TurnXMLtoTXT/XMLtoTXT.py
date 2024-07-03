'''This class object turns XMLs into TXT.

Author: Jerry Zou'''
import re, os, json, xml.etree.ElementTree as ET

class XMLtoTXT:
    def __init__ (self, folderPath1, folderPath2, infoAboutDesignatedFilesToUse, txtExportFolder, listOfXMLTagsToIgnore, listOfXMlDIVTypes):
        # "infoAboutDesignatedFilesToUse" has to be a JSON file path name. If the information about which specific files to iterate through is not a JSON file, change the code in the beginning of the "iterateFolder" function to accomodate.
        self.folderPath = [folderPath1, folderPath2] #Folder path name to whichever folder that holds the XML files.
        self.infoAboutDesignatedFilesToUse = infoAboutDesignatedFilesToUse
        self.txtExportFolder = txtExportFolder
        self.listOfXMLTagsToIgnore = listOfXMLTagsToIgnore
        self.designatedFilesToUse = [] #A list of file names in the folder that the user want to operate on. Could be empty.
        self.listOfXMlDIVTypes = listOfXMlDIVTypes

    def toTXT(self, originalPath):
        tree = ET.parse(originalPath)
        root = tree.getroot()

        def extractText(element):
            texts = []
            if element.tag not in self.listOfXMLTagsToIgnore and not (element.tag == "DIV1" and element.get("TYPE") in self.listOfXMlDIVTypes):
                if element.text:
                    texts.append(element.text.strip())
                for child in element:
                    texts.extend(extractText(child))
                if element.tail:
                    texts.append(element.tail.strip())
            return texts
        
        allText = extractText(root)
        combined = " ".join(allText)
        combined = combined.replace("\n", " ")
        combined = combined.replace("\r", " ")
        combined = " ".join(combined.split())
        return combined

    def iterateFolder(self):
        if self.infoAboutDesignatedFilesToUse: #This structure is specific to the use of a JSON file with the structure {"nameOfFile": placeHolder}. Change this structure according to the context of use.
            with open(self.infoAboutDesignatedFilesToUse, "r") as JSONFile:
                designatedFileNames = json.load(JSONFile)
            for name, placeHolder in designatedFileNames.items():
                self.designatedFilesToUse.append(name[:-4])
            
        allFileNames = []
        for folder in self.folderPath:
            for root, subFolder, files in os.walk(folder):
                for fileName in files:
                    if fileName.endswith(".xml") and not fileName.startswith("._"): # This ensures that only XML files are processed and not local metadata files about XMLs
                        if self.designatedFilesToUse: #If there are designated files to read (aka if the self.selectedFileNames list is NOT empty)
                            # FOR DEBUG print(fileName[:-7])
                            print(fileName[:-7])
                            if fileName[:-7] in self.designatedFilesToUse:
                            # for selectedXMLs in self.designatedFilesToUse: OBSOLETE CODE, Replaced by "if fileName[:-7] in self.selectedFileNames:"
                                # FOR DEBUG print(f"Processing {fileName}")
                                # if selectedXMLs in fileName: # only reads designated files. OBSOLETE CODE, Replaced by "if fileName[:-7] in self.designatedFilesToUse:"
                                fullFilePathName = os.path.join(root, fileName)
                                # FOR DEBUG print(fullFilePathName)
                                allFileNames.append(fullFilePathName)
                        else: # read all XML files in the folder when there is no designated files to read
                            fullFilePathName = os.path.join(root, fileName)
                            allFileNames.append(fullFilePathName)
        # print(allFileNames)

        for filePath in allFileNames:
            inTextFormat = self.toTXT(filePath)
            newTXTFilePath = self.txtExportFolder + "/" + filePath.split("/")[-1][:-7] + ".txt"
            with open(newTXTFilePath, "w") as newTXT:
                newTXT.write(inTextFormat)

if __name__ == "__main__":
    folderPath1 = "/Volumes/JZ/EEBOData+2024/Original Michigan XMLs/eebo_phase1/P4_XML_TCP"
    folderPath2 = "/Volumes/JZ/EEBOData+2024/Original Michigan XMLs/eebo_phase2/P4_XML_TCP_Ph2"
    infoOnDesignatedJSONFiles = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/XMLProcessingAndTraining/TurnXMLtoTXT/unique_filenames.json"
    txtExportFolder = "/Volumes/JZ/EEBOData+2024/Original Michigan XMLs/ExportTXT"
    listOfXMLTagsToIgnore = ["IDNO", "AVAILABILITY", "EXTENT", "PUBLICATIONSTMT", "SERIESSTMT", "NOTESSTMT", "ENCODINGDESC", "PROJECTDESC", "EDITORIALDECL", "PROFILEDESC", "REVISIONDESC", "CHANGE", "IDG", "STC", "VID"]
    listOfXMlDIVTypes = ["title page", "table of contents", "dedication"]
    XMLtoTXTObject = XMLtoTXT(folderPath1, folderPath2, infoOnDesignatedJSONFiles, txtExportFolder, listOfXMLTagsToIgnore, listOfXMlDIVTypes)
    XMLtoTXTObject.iterateFolder()