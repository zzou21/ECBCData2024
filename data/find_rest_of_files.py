import json

# Load the first JSON file (e.g., filenames1.json)
with open('/Users/lucasma/Desktop/filenames.json', 'r') as file:
    filenames1 = json.load(file)

# Load the second JSON file (e.g., filenames2.json)
with open('/Users/lucasma/Downloads/FilesBetween15901639.json', 'r') as file:
    filenames2 = json.load(file)

# Find filenames in the second list but not in the first
unique_filenames = {filename: 1 for filename in filenames2 if filename not in filenames1}


# Define the output JSON file path
output_file = '/Users/lucasma/Desktop/unique_filenames.json'

# Write the unique filenames to the new JSON file
with open(output_file, 'w') as json_file:
    json.dump(unique_filenames, json_file, indent=4)

print(f"Unique filenames have been stored in {output_file}")
