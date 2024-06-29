'''This file implements the Bible-Non-Bible Comparator fine-tuned MacBERTh to detect which verse is and is not Bible verse.

This file only contains one Python class object with no output methods as it is meant for other files to import this file and call on this Python class object to perform work.

This class object is already contained in the "IDandFilterVersesMain.py" file and is stored here again for back-up and testing purposes.

Author: Jerry Zou'''

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BibleVerseComparison:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        self.label_mapping = {0: False, 1: True}  # Adjust based on your label definitions. For example: self.label_mapping = {0: "Not Bible verse", 1: "Bible verse"}

    def checkBibleVerse(self, sentence):
        inputs = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=256, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
        return self.label_mapping[prediction]

'''---comment out the following lines when calling the class object from another file---'''

tunedModelPath = "/Users/Jerry/Desktop/ShuffleFullBibleModel" #refer to the fine-tuned model. Change this pathname according to your specific use
comparisonMachine = BibleVerseComparison(tunedModelPath)
# Exact verse from Geneva Bible: "And God made the beast of the earth according to his kinde, and the cattell according to his kinde, & euery creeping thing of the earth according to his kind: & God saw that it was good."
testSentence = "God mathe the beajt ov the ears , and the  according to his kinde, & euery creeeping thingss of the earths according to hiththththt kind: & God thaww that it wath jood."
print(f"'{testSentence}' -> {comparisonMachine.checkBibleVerse(testSentence)}")