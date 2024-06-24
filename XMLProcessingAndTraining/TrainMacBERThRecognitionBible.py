'''Fine-tuning MacBERTh on Geneva Bible and non-Geneva Bible sentences. Python Version. See other file for ipynb version. When using thsi code in DCC or other virtual machines, remember to change "csvFilePath" and "outputDir" variables so they point at the right file references.

Author: Jerry Zou'''

import pandas as pd, torch, os
from torch.utils.data import DataLoader, Dataset
# from transformers import AutoTokenizer AutoModelForSequenceClassification, AdamW
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sentence_transformers.models import Transformer, Pooling
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from transformers import TrainingArguments, Trainer

# Load the CSV into a DataFrame
csvFilePath = "/Users/Jerry/Desktop/FinalFullGenevaBibleRecognitionDataSetShuffled.csv"
dataFrame = pd.read_csv(csvFilePath)

# Convert DataFrame to a list of InputExamples
examples = []
for i, row in dataFrame.iterrows():
    examples.append(InputExample(texts=[row['Text']], label=float(row['Label'])))

# Split the data into train and validation sets
trainingExample, validationExample = train_test_split(examples, test_size=0.2, random_state=42)

# Define a custom dataset
class BibleDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        inputs = self.tokenizer(example.texts[0], padding='max_length', truncation=True, max_length=256, return_tensors='pt')
        inputs = {key: val.squeeze() for key, val in inputs.items()}
        inputs['labels'] = torch.tensor(example.label, dtype=torch.long)
        print(f"Index: {idx}, Keys: {list(inputs.keys())}, Shapes: {[val.shape for val in inputs.values()]}")
        inputs['labels'] = torch.tensor([example.label], dtype=torch.long)
        return inputs

# Load the tokenizer
word_embedding_model = models.Transformer('emanjavacas/MacBERTh')
tokenizer = word_embedding_model.tokenizer

# Create the datasets
train_dataset = BibleDataset(trainingExample, tokenizer)
val_dataset = BibleDataset(validationExample, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="/Users/Jerry/Desktop",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="/Users/Jerry/Desktop",
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Define a custom trainer to include SoftmaxLoss
class CustomTrainer(Trainer):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.loss_fct = losses.SoftmaxLoss(
            model=model,
            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
            num_labels=2
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").squeeze()  # Ensure labels are squeezed to match the expected shape
        outputs = model(**inputs)
        embeddings = outputs[0]
        loss = self.loss_fct(embeddings, labels)
        return (loss, outputs) if return_outputs else loss

# Initialize the SentenceTransformer model
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Initialize the CustomTrainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
outputDirectory = "/Users/Jerry/Desktop"
model.save(outputDirectory)
print("Training complete. Model saved to", outputDirectory)


'''

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
'''