from deepmultilingualpunctuation import PunctuationModel
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# Function to process a single file
def process_file(file_path):
    try:
        with open(file_path, "r") as f:
            text = f.read()
            print(f"File read successfully: {file_path}")  # Debugging print
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

        output_path = os.path.join(os.path.dirname(file_path), "PunctuationRestoration", os.path.basename(file_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            f.write(punctuated_text)

        print(f"Processed and saved: {file_path}")

# Main function to process files in parallel
def main():
    cwd = os.path.dirname(os.path.abspath(__file__))
    document_directory = os.path.join(cwd, "../dedication_text_EPcorpus_1590-1639")

    files_to_process = [os.path.join(document_directory, file_name) for file_name in os.listdir(document_directory) if file_name[0] != "."]
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_file, file_path) for file_path in files_to_process]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred during file processing: {e}")

if __name__ == "__main__":
    main()
