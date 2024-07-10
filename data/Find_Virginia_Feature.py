import os
import re
import shutil

# Define the directory containing the .txt files
source_directory = '/Users/lucasma/Downloads/Everything'

# Define the destination directory for matching files
destination_directory = '/Users/lucasma/Downloads/AllVirginia'

# Create a list of words to search for
words_to_search = ['virginia', 'uirginia', 'jamestown', 'james town', 'powhatan', 'pocahontas', 'rolfe', 'thomas dale','thomas gates', 'new world', 'college land']

# Compile a regular expression pattern for the words
pattern = re.compile('|'.join(words_to_search), re.IGNORECASE)

# List to store the names of files that contain any of the words
matching_files = []

# Ensure the destination directory exists
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Iterate over each file in the source directory
for filename in os.listdir(source_directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(source_directory, filename)
        
        # Read the content of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Check if any of the words are in the content
            if pattern.search(content):
                matching_files.append(filename)
                # Copy the file to the destination directory
                shutil.copy(file_path, os.path.join(destination_directory, filename))





