from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
import torch

# load the balanced dataset
file_path = 'balanced_reviews_sample.csv'  
df = pd.read_csv(file_path)

# prepare the dataset
class ReviewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text, 
            None, 
            add_special_tokens=True, 
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

MAX_LEN = 128
BATCH_SIZE = 16
train_texts, val_texts, train_labels, val_labels = train_test_split(df['Text'], df['Label'], test_size=0.1)
train_dataset = ReviewsDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, MAX_LEN)
val_dataset = ReviewsDataset(val_texts.tolist(), val_labels.tolist(), tokenizer, MAX_LEN)

# load the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy='steps',  # evaluate the model at each logging step
    logging_dir='./logs',
)

# define compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# train the model
trainer.train()

# evaluate the model
eval_result = trainer.evaluate()

# print the evaluation result
print(eval_result)