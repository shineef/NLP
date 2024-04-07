from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
import torch

from peft import LoraConfig, LoraModel, get_peft_model

# load the balanced dataset
file_path = 'balanced_reviews_sample.csv'
df = pd.read_csv(file_path)
# print(df.head())

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
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

MAX_LEN = 128
BATCH_SIZE = 16
train_texts, val_texts, train_labels, val_labels = train_test_split(df['Text'], df['Label'], test_size=0.1)
train_dataset = ReviewsDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, MAX_LEN)
val_dataset = ReviewsDataset(val_texts.tolist(), val_labels.tolist(), tokenizer, MAX_LEN)

# example = train_dataset[0]
# print(example)

# create a LoraConfig object
lora_config = LoraConfig(
    r=4,
    lora_alpha=1,
    lora_dropout = 0.1,
    use_rslora = True,
    target_modules='all-linear'
)

# load bert model and apply LORA
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model = get_peft_model(model, lora_config, adapter_name='default')

# class CustomTrainer(Trainer):
#     def _save_checkpoint(self, model, trial, metrics=None):
#         if 'eval_loss' in metrics:
#             super()._save_checkpoint(model, trial, metrics)

class CustomTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        if metrics is None or 'eval_loss' not in metrics:
            super()._save_checkpoint(model, trial)
        else:
            super()._save_checkpoint(model, trial, metrics)

    def training_step(self, model, inputs):
        # call the original training_step method
        loss = super().training_step(model, inputs)

        # compute the metrics and add them to logs every 500 steps
        if self.state.global_step % 500 == 0:
            labels = inputs["labels"]
            preds = model(**inputs).logits.argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu(), preds.cpu(), average='binary')
            acc = accuracy_score(labels.cpu(), preds.cpu())
            metrics = {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
            self.log(metrics)

        return loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # call the original evaluate method
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # compute the metrics directly in the evaluate method
        if isinstance(output, EvalPrediction):
            labels = output.label_ids
            preds = output.predictions.argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
            acc = accuracy_score(labels, preds)
            metrics = {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
            for key in list(metrics.keys()):
                if not key.startswith(metric_key_prefix):
                    metrics[metric_key_prefix + "_" + key] = metrics.pop(key)
            return metrics, output

# define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy='steps',  # evaluate the model at each logging step
    # logging_steps=10,
    logging_dir='./logs',
    load_best_model_at_end=True,
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
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# # obtain the data loader
# train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Use a small batch size for testing

# # obtain the batch
# batch = next(iter(train_dataloader))

# # check the batch
# print(batch)

# # print the length of the datasets
# print(len(train_dataset))
# print(len(val_dataset))

# train the model
trainer.train()

# evaluate the model
eval_result = trainer.evaluate()

# print the evaluation result
print(eval_result)
