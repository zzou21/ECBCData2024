import os
import json
import shutil

# Load the file list from 600File.json
with open("FindCosineSimilarityWords/600Files.json", "r") as f:
    file_list = json.load(f)

# Define the source and destination directories
document_directory = "/Users/lucasma/Downloads/dedication_text_EPcorpus_1590-1639"
destination_directory = "/Users/lucasma/Downloads/AllVA_cleaned"

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# Move the files from the document directory to the destination directory
for file_name in file_list:
    source_path = os.path.join(document_directory, file_name+".txt")
    if os.path.exists(source_path):
        destination_path = os.path.join(destination_directory, file_name+".txt")
        shutil.move(source_path, destination_path)
        print(f"Moved: {file_name}")
    else:
        print(f"File not found: {file_name}")
