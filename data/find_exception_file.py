import os
import json

# Define the directory
directory1 = "/Users/lucasma/Downloads/EEBOphaseI_1590-1639_body_texts"
directory2 = "/Users/lucasma/Downloads/EEBOphase2_1590-1639_body_texts"

# Initialize a dictionary to store filenames
file_dict = {}

# Iterate over all files in the directory
for filename in os.listdir(directory1):
    # Add the filename to the dictionary with value 1
    if filename[0] != ".":
        file_dict[filename] = 1

for filename in os.listdir(directory2):
    # Add the filename to the dictionary with value 1
    if filename[0] != ".":
        file_dict[filename] = 1

# Define the output JSON file path
output_file = "/Users/lucasma/Desktop/filenames.json"

# Write the dictionary to the JSON file
with open(output_file, "w") as json_file:
    json.dump(file_dict, json_file, indent=4)

print(f"Filenames have been stored in {output_file}")
