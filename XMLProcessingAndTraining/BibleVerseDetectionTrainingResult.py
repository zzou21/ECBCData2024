'''This file implements the Bible-Non-Bible Comparator fine-tuned MacBERTh to detect which verse is and is not Bible verse.

This file only contains one Python class object with no output methods as it is meant for other files to import this file and call on this Python class object to perform work.

Author: Jerry Zou'''
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BibleVerseClassifier:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        self.label_mapping = {0: "Not a Bible Verse", 1: "Bible Verse"}  # Adjust based on your label definitions

    def checkBibleVerse(self, sentence):
        inputs = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=256, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
        
        return self.label_mapping[prediction]
    
tunedModelPath = "/Users/Jerry/Desktop/100Paramtest" #refer to the fine-tuned model. Change this pathname according to your specific use
comparisonMachine = BibleVerseClassifier(tunedModelPath)

example_sentence = "We will fight them on the beaches."
print(f"'{example_sentence}' -> {comparisonMachine.checkBibleVerse(example_sentence)}")
