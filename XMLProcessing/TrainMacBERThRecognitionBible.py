'''Fine-tuning MacBERTh on Geneva Bible and non-Geneva Bible sentences. Python Version. See other file for ipynb version.

Author: Jerry Zou'''

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# csvFilePath = "/Users/Jerry/Desktop/GenevaBibleRecognitionDataSetUpdated.csv"
csvFilePath = "/Users/Jerry/Desktop/SmallDataGenevaBibleRecognitionDataSetUpdated.csv"
#When using DCC:
# csvFilePath = "/hpc/group/datap2023ecbc/zz341/GenevaBibleRecognitionDataSetUpdated.csv"
# csvFilePath = "/hpc/group/datap2023ecbc/zz341/SmallDataGenevaBibleRecognitionDataSetUpdated.csv"
dataFrame = pd.read_csv(csvFilePath)

hf_dataset = HFDataset.from_pandas(dataFrame)
tokenizer = AutoTokenizer.from_pretrained('emanjavacas/MacBERTh')
model = AutoModelForSequenceClassification.from_pretrained('emanjavacas/MacBERTh', num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples['Text'], padding='max_length', truncation=True, max_length=256)

tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)

tokenized_datasets = tokenized_datasets.rename_column('Label', 'labels')
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
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

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

progress_bar = tqdm(range(num_epochs * len(train_loader))) # set up training loop

model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)

output_dir = '/Users/Jerry/Desktop'
# When using DCC:
# output_dir = "/hpc/group/datap2023ecbc/zz341"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Training complete. Model saved to", output_dir)