import pandas as pd, json
import os
class overlayMetadataToCSV:
    def __init__ (self, txtRawPath, outputCSVPath, columnNames, metadataJSON):
        self.txtRawPath = txtRawPath
        self.outputCSVPath = outputCSVPath
        self.columnNames = columnNames
        self.metadataJSON = metadataJSON

    def accessMetadataJSON(self):
        with open(self.metadataJSON, "r") as jsonFile:
            content = json.load(jsonFile)
        return content

    def accessSpecificMetadata(self, fileName):
        metadataContent = self.accessMetadataJSON() # A dictionary, since it is directly returned from a JSON
        intendedMetadataDict = None
        for phase, fileNameDictionary in metadataContent.items():
            if fileName in fileNameDictionary:
                intendedMetadataDict = fileNameDictionary[fileName]
        returnList = [intendedMetadataDict["TITLE"], intendedMetadataDict["AUTHOR"], intendedMetadataDict["DATE"]]
        return returnList

    def toCSV(self):
        with open(self.txtRawPath) as txtContent:
            content = txtContent.read()
        contentList = [data for data in content.split("\n") if data]
        contentListList = [data.split(": ") for data in contentList]
        contentListList = [[name[:-4], dataList[1:-1]] for name, dataList in contentListList] #after this, a list has 4 indexes, the first being file name, 2-4 being projection scores.
        contentListList = [[name, [float(score) for score in dataList.split(", ")]] for name, dataList in contentListList]
        # [["A00000", [0.0, 0.0, 0.0]], ["A00000", [0.0, 0.0, 0.0]]]
        # contentListList = [[name, [float(score) for score in dataList]] for name, dataList in contentListList]

        def flattenList(listToFlatten):
            flatList = []

            #recursion
            def flatten(sublist):
                for item in sublist:
                    if isinstance(item, type(list)):flatten(item)
                    else: flatList.append(item)
            
            flatten(listToFlatten)
            return flatList
        
        combinedList = []

        dataFrameInit = {}
        for list in contentListList:
            newList = flattenList(list)
            listWithMetadata = newList + self.accessSpecificMetadata(list[0])
            combinedList.append(listWithMetadata)
        for index in range(len(self.columnNames)):
            dataFrameInit[self.columnNames[index]] = [names[index] for names in combinedList]
        newDataFrame = pd.DataFrame(dataFrameInit)
        newDataFrame.to_csv(self.outputCSVPath, index=False)

if __name__ == "__main__":
    wd = os.getcwd()
    txtRawPath = os.path.join(wd, "data/projection_result_G3_NT.txt")
    outputCSVPath = os.path.join(wd, "data/projectionResultWithMetadata_G3_NT.csv")
    columnNames = ["File Name", "Projection #1", "Projection #2", "Projection #3", "Manuscript Title", "Author", "Publication Year"]
    metadataJSON = os.path.join(wd, "./XMLProcessingAndTraining/ManuscriptMetadata/cleanedDocumentMetadata.json")
    overlay = overlayMetadataToCSV(txtRawPath, outputCSVPath, columnNames, metadataJSON)
    overlay.toCSV()