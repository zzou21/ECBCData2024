import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# Load the CSV into a DataFrame
csvFilePath = "/hpc/group/datap2023ecbc/zz341/GenevaBibleRecognitionDataSetUpdated.csv"
dataFrame = pd.read_csv(csvFilePath)

# Convert DataFrame to Hugging Face Dataset
hf_dataset = HFDataset.from_pandas(dataFrame)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('emanjavacas/MacBERTh')

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['Text'], padding='max_length', truncation=True, max_length=256)

tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)

# Set the format of the dataset to return PyTorch tensors
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Shuffle the dataset before splitting
tokenized_datasets = tokenized_datasets.shuffle(seed=42)

# Split the dataset into train and test
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_data = train_test_split['train']
val_data = train_test_split['test']

# Define the custom PyTorch Dataset class
class BibleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        item = self.data[idx]
        return {key: val.clone().detach() if torch.is_tensor(val) else torch.tensor(val) for key, val in item.items()}

    def __len__(self):
        return len(self.data)

# Create the custom datasets
train_dataset = BibleDataset(train_data)
val_dataset = BibleDataset(val_data)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Load the pre-trained model for masked language modeling
model = AutoModelForMaskedLM.from_pretrained('emanjavacas/MacBERTh')

# Data collator for masked language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15
)

# Training arguments
training_args = TrainingArguments(
    output_dir="/hpc/group/datap2023ecbc/zz341/finetunedBibleNonBible",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
trainer.save_model("/hpc/group/datap2023ecbc/zz341/finetunedBibleNonBible")
tokenizer.save_pretrained("/hpc/group/datap2023ecbc/zz341/finetunedBibleNonBible")

print("Training complete. Model saved to /hpc/group/datap2023ecbc/zz341/finetunedBibleNonBible")
