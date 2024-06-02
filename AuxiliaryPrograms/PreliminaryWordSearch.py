'''Counting how many times a word, or a selection of words, appears in selected EEBO documents. User could customize the code to process either a single txt file or a folder of txt files.

Author: Jerry Zou

Places in the code that's customizable:
- Path name for single file processing -> go to main() method and change String variable "pathNameForSingleFile".
- Path name for folder of files processing -> go to main() method and change String variable "pathNameForFolderOfFiles".
- Words to search -> go to main() method and change List variable "findTheseWordsPlease"
- Printing out how many names of most occuring files -> go to Line 93 and change the iteration number.
- Printing out the line of the file in which the word to search appears in -> run the code until the system asks for input.

To debug code, un-comment the lines that start with "FOR DEBUG."
'''
import os

class wordAppearanceLocationCalculator:
    def __init__ (self, sourceSingleFile, sourceFolder, wordToFind):
        self.sourceSingleFile = sourceSingleFile
        self.sourceFolder = sourceFolder
        self.wordToFind = wordToFind

    def wordCounterForSingleFiles(self, sourceSingleFile, wordToFind):
        counter = 0
        #This dictionary (a key:value pair of string:integer) contains the line in the document where the wordToFind words appeared. The key is the content of the line; the value is the line number:
        locationOfAppearance = {}
        #FOR DEBUG: temporaryList = []
        print("Processing file...")
        with open(sourceSingleFile, "r") as fileVariable:
            allLines = fileVariable.readlines()
            for indexPosition, lineContent in enumerate(allLines, start=1):
                #FOR DEBUG: print(indexPosition)
                #FOR DEBUG: temporaryList.append(indexPosition)
                lineContent = lineContent.lower().rstrip("\n")
                for word in wordToFind:
                    if word in lineContent:
                        counter += 1
                        print(f"Line {lineContent} contains {word}")
                        locationOfAppearance[lineContent] = indexPosition
                    else:
                        print(f"Line {lineContent} does not contain {word}")
            #FOR DEBUG: print(temporaryList[:100])
        print("Processing completed.")
        print(f"Words in {wordToFind} appeared in the file for a total of {counter} times.")

        '''For the two lines below, un-comment the line that you wish to see content in. The first prints the numerical location of the word to find (i.e.: the "value" of the "locationOfAppearance" dictionary, or the line count in the document). The second prints out the actual content of that line (i.e.: the "key" of the "locationOfAppearance" dictionary.)'''
        #print(f"Words in {wordToFind} appeared in lines {', '.join(str(ind)for ind in locationOfAppearance.values())} of the file.")
        print(f"Word in {wordToFind} appeared in these lines: {locationOfAppearance.keys()}")

    def wordCounterForFolderOfFiles(self, sourceFolder, wordToFind):
        counter = 0
        #The key of this dictionary's key:value pairs (list:integer) is a list with the 0th index holding the file name and the 1st index is the line where the word appears in; the value is the line number of the file in which the word appears in
        locationOfAppearance = {}
        print("Processing folder...")
        for filename in os.listdir(sourceFolder):
            if filename.endswith(".txt"): # This checks if the file is a txt file and not another subfolder or file of another format
                filePath = os.path.join(sourceFolder, filename)
                with open(filePath, "r") as fileToOpen:
                    allLines = fileToOpen.readlines()
                    for indexPosition, lineContent in enumerate(allLines, start=1):
                        #FOR DEBUG: print(indexPosition)
                        #FOR DEBUG: temporaryList.append(indexPosition)
                        lineContent = lineContent.lower().rstrip("\n")
                        for word in wordToFind:
                            if word in lineContent:
                                counter += 1
                                '''Commented the line below because DO NOT print if the folder is large, otherwise it will take FOREVER to print. Un-comment if you really want to see every single line printed.'''
                                #print(f"Line {lineContent} in file {filePath} contains {word}")
                                locationOfAppearance[filename, lineContent] = indexPosition
                            #else:
                                '''Same reason as above for commenting out the line below.'''
                                #print(f"Line {lineContent} in file {filePath} does not contain {word}")
        print("Processing completed.")
        print(f"Words in {wordToFind} appeared in the folder for a total of {counter} times.")
        #the dictionary below (key:value pair in the type string:integer) stores how many times the words to search for have appeared in each of the files. E.g. "ViCo.txt" : 4 means the words to search appeared 4 times in the file "ViCo.txt".
        dictionaryOfFileAppearances = {}
        for key, value in locationOfAppearance.items():
            #as a reminder, key[0] is file name, key[1] is content of a line, and value is the line number of key[1] inside the file key[0].
            if key[0] not in dictionaryOfFileAppearances:
                dictionaryOfFileAppearances[key[0]] = 0
            dictionaryOfFileAppearances[key[0]] += 1

        # this sorts the dictionaryOfFileAppearances into descending order, from the file with most appearances to least appearances. This does not include files that have 0 appearances.
        sortedDictionaryOfFileAppearances = {key: value for key, value in sorted(dictionaryOfFileAppearances.items(), key = lambda appearanceAmountForEachFile : appearanceAmountForEachFile[1], reverse = True)}
        
        '''FOR DEBUG:
        counterTemporary = 0
        for k, v in sortedDictionaryOfFileAppearances.items():
            counterTemporary += v
        print(f"Total times: {str(counterTemporary)}")'''

        printLines = input(f"Do you want to print out the lines where the words to search appeared in? (there will be {len(locationOfAppearance.keys())} lines be printed if you select 'Y') (Y/N)")
        if printLines == "Y":
            for list in locationOfAppearance.keys():
                print(f"In file {list[0]}: {list[1]}\n")

        print("In descending order, these 10 files contain the most amount of words to search: ")
        for numberOfFiles, (fileContainingWord, numberOfAppearance) in enumerate(sortedDictionaryOfFileAppearances.items()):
            if numberOfFiles < 10: #change this iteration number to how many top appearance files you wish to print.
                print(f"The words to search appeared in {fileContainingWord} {numberOfAppearance} times.")
            else:
                break

if __name__ == "__main__":
    # Put the path name here if you are calculating from a SINGLE FILE
    pathNameForSingleFile = "/Users/Jerry/Desktop/ViCocombined.txt"
    # Put the path name here if you are calculating from a FOLDER OF FILES 
    pathNameForFolderOfFiles = "/Users/Jerry/Desktop/EEBOphase2_1590-1639_body_texts"
    #Write the words to count in the "findTheseWordsPlease" list variable:
    findTheseWordsPlease = ["thrope", "thorpe", "george thrope", "george thorpe"]
    #Creating python object
    wordAppearanceObject = wordAppearanceLocationCalculator(pathNameForSingleFile, pathNameForFolderOfFiles, findTheseWordsPlease)
    '''Un-comment the function you wish to run to prevent confusion when both functions are running simultaeously: '''
    #wordAppearanceObject.wordCounterForSingleFiles(pathNameForSingleFile, pathNameForFolderOfFiles, findTheseWordsPlease)
    wordAppearanceObject.wordCounterForFolderOfFiles(pathNameForFolderOfFiles, findTheseWordsPlease)