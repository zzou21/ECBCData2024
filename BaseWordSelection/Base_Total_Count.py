import os

def count_files_with_keyword(directory, keyword):
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".txt") and filename[0] != ".":
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                content = content.lower()
                if keyword in content:
                    count += 1
    return count

# Specify the directory and keyword
directory = "/Users/lucasma/Downloads/AllVirginia"
keyword = "native"

# Get the count of files containing the keyword
count = count_files_with_keyword(directory, keyword)
print(f"Number of .txt files containing the keyword '{keyword}': {count}")
