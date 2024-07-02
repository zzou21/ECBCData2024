import os
import re
import chardet

def count_documents_with_word(folder_path, word):
    count = 0
    word_pattern = re.compile(rf'\b{word}\b', re.IGNORECASE)
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") and filename[0] != ".":  # Consider only text files
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                try:
                    content = raw_data.decode(encoding)
                    if word_pattern.search(content):
                        count += 1
                except (UnicodeDecodeError, TypeError) as e:
                    print(f"Error decoding file {filename}: {e}")

    return count

# Example usage
folder_path = '/Volumes/JZ/EEBOData+2024/EEBOphaseI_1590-1639_body_texts'
word = 'uirginia'
documents_count = count_documents_with_word(folder_path, word)
print(f"The word '{word}' appears in {documents_count} documents.")
