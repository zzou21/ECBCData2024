import re, os, xml.etree.ElementTree as ET

def XMLtoTXT(originalPath):
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

def iterateFolder(folderPath):
    allFileNames = []
    for folder in folderPath:
        for root, subFolder, files in os.walk(folder):
            for fileName in files:
                if fileName.endswith(".xml") and not fileName.startswith("._"): # This ensures that only XML files are processed and not local metadata files about XMLs
                    if selectedFileNames: #If there are designated files to read (aka if the self.selectedFileNames list is NOT empty)
                        # FOR DEBUG print(fileName[:-7])
                        if fileName[:-7] in self.selectedFileNames:
                        # for selectedXMLs in self.selectedFileNames: OBSOLETE CODE, Replaced by "if fileName[:-7] in self.selectedFileNames:"
                            # FOR DEBUG print(f"Processing {fileName}")
                            # if selectedXMLs in fileName: # only reads designated files. OBSOLETE CODE, Replaced by "if fileName[:-7] in self.selectedFileNames:"
                            fullFilePathName = os.path.join(root, fileName)
                            # FOR DEBUG print(fullFilePathName)
                            allFileNames.append(fullFilePathName)
                    else: # read all XML files in the folder when there is no designated files to read
                        fullFilePathName = os.path.join(root, fileName)
                        allFileNames.append(fullFilePathName)



if __name__ == "__main__":
    folderPath = "/Users/Jerry/Desktop/A0"
    originalPath = "/Users/Jerry/Desktop/A0/A00002.P4.xml"
    TXTPath = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/XMLProcessingAndTraining/TurnXMLtoTXT/test1.txt"
    text = XMLtoTXT(originalPath)
    with open(TXTPath, "w") as TXTfile:
        TXTfile.write(text)