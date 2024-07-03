txtExportFolder = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/XMLProcessingAndTraining/TurnXMLtoTXT/ExportedTXT"
a = '/Volumes/JZ/EEBOData+2024/Original Michigan XMLs/eebo_phase2/P4_XML_TCP_Ph2/A8/A86742.P4.xml'
b = txtExportFolder + "/" + a.split("/")[-1][:-7] + ".txt"
print(b)