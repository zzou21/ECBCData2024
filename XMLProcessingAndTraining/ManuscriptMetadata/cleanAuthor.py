import re, json, os

def cleanAuthor(jsonPath):
    with open(jsonPath, "r") as jsonContent:
            metadataContent = json.load(jsonContent)

    for phase, filenameDictionary in metadataContent.items():
            for filename, metadataDictionary in filenameDictionary.items():
                
                if metadataDictionary["AUTHOR"]:
                    
                    for author in metadataDictionary["AUTHOR"]:
                        
                        # Rough cleaning with regular expression. Take first 2 parts as separated by comma.
                        name_pattern = re.compile(r'^([^,]+, [^,]+)')
                        match = name_pattern.match(author)
    
                        if match:
                            name_part = match.group(0)
                        else:
                            name_part = author


                        # Clean ".fl" which is usually followed by a year range
                        fl_index = name_part.find(' fl. ')
                        if fl_index != -1:
                            # Return the part of the entry before ".fl"
                            cleaned_entry = name_part[:fl_index].strip()
                            name_part = cleaned_entry

                        # Clean ", d." which is usually followed by a year range
                        d_index = name_part.find(', d.')
                        if d_index != -1:
                            # Return the part of the entry before ".fl"
                            cleaned_entry = name_part[:d_index].strip()
                            name_part = cleaned_entry


                        with open(os.path.join(currentDir, "test.txt"), "a") as f:
                            print(name_part, file=f)
                    






currentDir = os.path.dirname(os.path.abspath(__file__))
jsonPath = os.path.join(currentDir, "[Clean]DocumentMetadata.json")
cleanAuthor(jsonPath)