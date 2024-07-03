import re, os, json, xml.etree.ElementTree as ET

class XMLtoTXT:
    def __init__ (self, folderPath1, folderPath2, infoAboutDesignatedFilesToUse, txtExportFolder):
        # "infoAboutDesignatedFilesToUse" has to be a JSON file path name. If the information about which specific files to iterate through is not a JSON file, change the code in the beginning of the "iterateFolder" function to accomodate.
        self.folderPath = [folderPath1, folderPath2] #Folder path name to whichever folder that holds the XML files.
        self.infoAboutDesignatedFilesToUse = infoAboutDesignatedFilesToUse
        self.designatedFilesToUse = [] #A list of file names in the folder that the user want to operate on. Could be empty.
        # self.AllFileNamesToOperate = [] #Holds all file names to operate on

    def XMLtoTXT(self, originalPath):
        tree = ET.parse(originalPath)
        root = tree.getroot()

        def extractText(element):
            texts = []
            if element.text:
                texts.append(element.text.strip())
            for child in element:
                texts.extend(extractText(child))
            if element.tail:
                texts.append(element.tail.strip())
            return texts
        
        allText = extractText(root)
        combined = " ".join(allText)
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
            print(allFileNames)
            # for filePath in allFileNames:


if __name__ == "__main__":
    folderPath1 = "/Volumes/JZ/EEBOData+2024/Original Michigan XMLs/eebo_phase1/P4_XML_TCP"
    folderPath2 = "/Volumes/JZ/EEBOData+2024/Original Michigan XMLs/eebo_phase2/P4_XML_TCP_Ph2"
    # originalPath = "/Users/Jerry/Desktop/A0/A00002.P4.xml"
    # TXTPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/XMLProcessingAndTraining/TurnXMLtoTXT/test1.txt"
    infoOnDesignatedJSONFiles = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/XMLProcessingAndTraining/TurnXMLtoTXT/unique_filenames.json"
    txtExportFolder = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/XMLProcessingAndTraining/TurnXMLtoTXT/ExportedTXT"
    XMLtoTXTObject = XMLtoTXT(folderPath1, folderPath2, infoOnDesignatedJSONFiles, txtExportFolder)
    XMLtoTXTObject.iterateFolder()