from deepmultilingualpunctuation import PunctuationModel
import os, json

# Load the pre-trained model
model = PunctuationModel()

# Function to split text into chunks and restore punctuation
def process_text(text, chunk_size=128):
    words = text.split()
    punctuated_text = ""
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        punctuated_chunk = model.restore_punctuation(chunk)

        # Find the last sentence-ending punctuation
        last_period = max(punctuated_chunk.rfind('.'), punctuated_chunk.rfind('!'), punctuated_chunk.rfind('?'))
        
        if last_period != -1:
            # Adjust the chunk to end at the last period
            punctuated_text += punctuated_chunk[:last_period+1] + " "
            remaining_text = punctuated_chunk[last_period+1:].strip()
        else:
            punctuated_text += punctuated_chunk + " "
            remaining_text = ""

        # Adjust the original text to remove the processed part
        start += chunk_size

        if remaining_text:
            remaining_words = remaining_text.split()
            start -= len(remaining_words)

    return punctuated_text.strip()

# Example usage
document_directory = "/Users/lucasma/Downloads/80Files"
cwd = os.getcwd()

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
            punctuated_text = process_text(text)

            with open(os.path.join(cwd, "2023_cleaned", file_name), "a") as f:
                f.write(punctuated_text)

        else:
            print("No text to process.")