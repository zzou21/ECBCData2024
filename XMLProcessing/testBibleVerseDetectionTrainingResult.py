from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

traineModel = "/Users/Jerry/Desktop/BibleTestModel"
model = AutoModelForSequenceClassification.from_pretrained(traineModel)
tokenizer = AutoTokenizer.from_pretrained(traineModel)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()
def is_bible_verse(sentence):
    inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=256, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
    
    label_mapping = {0: "Not a Bible Verse", 1: "Bible Verse"}  # Adjust based on your label definitions
    return label_mapping[prediction]

exampleSentence = "In the beginning God created the heaven and the earth"
print(f"'{exampleSentence}' -> {is_bible_verse(exampleSentence)}")