'''This function, "accessXMLDictionaryTool", is a tool to traverse the dictionary format in JSON file in "StoringItalicsAndLineNumber.json". Designed to reuse and customize according to each occasion. Write the customized command to each of the phrases in the JSON file in designated space marked by "Write commands to the phrases below". 

Author: Jerry Zou'''

import json
def accessXMLDictionaryTool(jsonPath):
    with open(jsonPath, "r") as file:
        content = json.load(file)
    newDictionary = {} #builds a new dictionary after completing the customized computing and commands specified by the user below
    for pair in content:
        fileName = pair #grabs the file name, e.g. "A00001"
        # print(f"fileName : {fileName}")
        tagsDictionary = content[pair] #grabs the dictionary containing all tags
        # print(f"tagsDictionary: {tagsDictionary}")
        newDictionary[fileName] = [{}]
        for tagName, lineDictionary in tagsDictionary[0].items(): #iterate through tags dictionary
            newTag = {} #starts a new dictionary at the tag level
            for lineNumber, sentences in lineDictionary.items():
                newSentences = [] 
                for phrase in sentences: #base level. Access phrase content in this for loop.
                    ''' ------ Write commands to the phrases below ----- '''
                    # if len(phrase) >= 20:
                        # FOR DEBUG if "Tobacco" in phrase:
                    editedPhrase = phrase.lower() #EXAMPLE OPERATION, replace this with needed changes
                    newSentences.append(editedPhrase) #put the changed content back to the list
                    print(f"Computed sentence: {editedPhrase}")
                    ''' ------ End of command for phrases ----- '''
                if newSentences: # this If statement will control that only if, after editing, a line still with content will be added back to the new dictionary.
                    newTag[lineNumber] = newSentences
            if newTag : #this If statement ensures that only, after editing, a tag dictionary with content will be added back to the new dictionary.
                newDictionary[fileName][0][tagName] = newTag

if __name__ == "__main__":
    jsonPath = "XMLProcessingAndTraining/StoringItalicsAndLineNumber.json" #change pathname reference accordingly.
    accessXMLDictionaryTool(jsonPath)