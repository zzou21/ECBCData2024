'''Fine-tuning MacBERTh on Geneva Bible and non-Geneva Bible sentences. Python Version. See other file for ipynb version. When using thsi code in DCC or other virtual machines, remember to change "csvFilePath" and "outputDir" variables so they point at the right file references.

Author: Jerry Zou'''

import pandas as pd, torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# csvFilePath = "/Users/Jerry/Desktop/GenevaBibleRecognitionDataSetUpdated.csv"
#When using DCC:
csvFilePath = "/hpc/group/datap2023ecbc/zz341/GenevaBibleRecognitionDataSetUpdated.csv"

dataFrame = pd.read_csv(csvFilePath)

hfData = HFDataset.from_pandas(dataFrame)
tokenizer = AutoTokenizer.from_pretrained('emanjavacas/MacBERTh')
model = AutoModelForSequenceClassification.from_pretrained('emanjavacas/MacBERTh', num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples['Text'], padding='max_length', truncation=True, max_length=256)

tokenizedDatasets = hfData.map(tokenize_function, batched=True)

tokenizedDatasets = tokenizedDatasets.rename_column('Label', 'labels')
tokenizedDatasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

train_test_split = tokenizedDatasets.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']

class BibleDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        item = self.data[idx]
        return {key: val.clone().detach() if torch.is_tensor(val) else torch.tensor(val) for key, val in item.items()}
    def __len__(self):
        return len(self.data)

train_dataset = BibleDataset(train_dataset)
val_dataset = BibleDataset(val_dataset)

trainLoad = DataLoader(train_dataset, batch_size=8, shuffle=True)
validationLoad = DataLoader(val_dataset, batch_size=8, shuffle=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
epochCount = 3

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

progressIndicatorBar = tqdm(range(epochCount * len(trainLoad))) # set up training loop

model.train()
for epoch in range(epochCount):
    for batch in trainLoad:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progressIndicatorBar.update(1)

# outputDirectory = '/Users/Jerry/Desktop'
# When using DCC:
outputDirectory = "/hpc/group/datap2023ecbc/zz341/finetunedBibleNonBible"
model.save_pretrained(outputDirectory)
tokenizer.save_pretrained(outputDirectory)

print("Training complete. Model saved to", outputDirectory)