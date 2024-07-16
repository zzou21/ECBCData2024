from deepmultilingualpunctuation import PunctuationModel
import os

# Load the pre-trained model
model = PunctuationModel()

# Function to split text into chunks
def split_text(text, chunk_size=128, overlap=32):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Function to restore punctuation
def restore_punctuation(text):
    chunks = split_text(text)
    punctuated_chunks = [model.restore_punctuation(chunk) for chunk in chunks]
    punctuated_text = " ".join(punctuated_chunks)
    return punctuated_text

# Example usage
cwd = os.getcwd()

document_directory = "/Users/lucasma/Downloads/80Files"

for file_name in os.listdir(document_directory):
    file_path = os.path.join(document_directory, file_name)
    
    if file_path[0] != ".":
        try:
            with open(file_path, "r") as f:
                text = f.read()
                print("File read successfully.")  # Debugging print
        except FileNotFoundError:
            print(f"The file at {file_path} was not found.")
            text = ""
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            text = ""

        if text:
            print(f"Text length: {len(text)} characters")  # Debugging print
            text = text.lower()  # Optional: Lowercase text, consider if needed
            punctuated_text = restore_punctuation(text)

            with open(os.path.join(cwd, "2023_cleaned", file_name), "a") as f:
                f.write(punctuated_text)

        else:
            print("No text to process.")
