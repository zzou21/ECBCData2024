''' This python file is an interface that streamlines workflow so the user can record XML tags with ease and have these tags stored in an organized JSON file.

Author: Jerry Zou'''
import json
from datetime import datetime
def interface(jsonPathName):
    newTagsAsSet = set()
    tagDictionaryReadyForJsonDump = {}
    time = datetime.now()
    timeAsDictKey = time.strftime("%Y-%m-%d %H:%M:%S")
    print("Instructions:\nWhen typing the XML tag, do NOT add the carrot signs. Upload are case sensitive; please retain capitalization.\nE.g.: if an XML tag is <Tag> text <\Tag>, only type \"Tag\". NOT: \"tag\" or \"<Tag>\". To exit, type \"exit\".")

    while True:
        inputtedTag = ""
        userInput = input("Please enter the tag you like to upload: ")
        inputtedTag = userInput
        if inputtedTag == "exit":
            break
        newTagsAsSet.add(inputtedTag)

    with open(jsonPathName, "r") as file: existingTagsDictionary = json.load(file)

    #This is to check for any repetitions:
    newTagsList = list(newTagsAsSet)
    for tagList in existingTagsDictionary.values():
        for tag in tagList:
            if tag in newTagsList:
                newTagsList.remove(tag)

    tagDictionaryReadyForJsonDump[timeAsDictKey] = newTagsList
    existingTagsDictionary.update(tagDictionaryReadyForJsonDump)
    with open(jsonPathName, "w") as file: json.dump(existingTagsDictionary, file, indent=4)

if __name__ == "__main__":
    print("---XML Tag Update Interface---")
    jsonPath = "XMLProcessing/XMLCitationTags.json"
    interface(jsonPath)
    print("---Interface turned off---")