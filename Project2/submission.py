####################initial code####################
import pandas as pd
import matplotlib.pyplot as plt

# Load the Data
file_path = 'Reviews.csv'
reviews_df = pd.read_csv(file_path)

# Since we only use scores and text, lets see score distribution for this part
score_distribution = reviews_df['Score'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
score_distribution.plot(kind='bar')
plt.title('Distribution of Review Scores')
plt.xlabel('Score')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.savefig('score_distribution.png')
plt.show()

# Label Creation
reviews_filtered_df = reviews_df[reviews_df['Score'] != 3]

# Create a new column for labels: 1 for positive (score > 3) and 0 for negative (score < 3)
reviews_filtered_df['Label'] = (reviews_filtered_df['Score'] > 3).astype(int)

# Check the balance of the newly created labels
label_distribution = reviews_filtered_df['Label'].value_counts(normalize=True)
print(label_distribution)

# Separate positive and negative reviews
positive_reviews = reviews_filtered_df[reviews_filtered_df['Label'] == 1]
negative_reviews = reviews_filtered_df[reviews_filtered_df['Label'] == 0]

# determine the sample size
total_count = len(reviews_filtered_df)
target_sample_size = int(total_count * 0.2)  
negative_count = len(negative_reviews)

if negative_count >= target_sample_size / 2:
    positive_sample = positive_reviews.sample(n=target_sample_size // 2, random_state=42)
    negative_sample = negative_reviews.sample(n=target_sample_size // 2, random_state=42)
else:
    negative_sample = negative_reviews
    positive_sample = positive_reviews.sample(n=negative_count, random_state=42)

# combine the samples to form a balanced dataset
balanced_samples = pd.concat([positive_sample, negative_sample])

total_balanced_samples = balanced_samples.shape[0]
total_original_samples = reviews_df.shape[0]
proportion_balanced_to_original = total_balanced_samples / total_original_samples

save_path = 'balanced_reviews_sample.csv'
balanced_sample_file_path = balanced_samples.to_csv(save_path, index=False)

# Print statements to see the values (you can comment these out if needed)
print(f"Total samples in balanced dataset: {total_balanced_samples}")
print(f"Proportion of balanced dataset to original: {proportion_balanced_to_original}")
print(f"Balanced dataset saved to: {balanced_sample_file_path}")

####################preprocess code####################
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup

# read the balanced dataset 
df = pd.read_csv('balanced_reviews_sample.csv')

stop_words = set(stopwords.words('english'))

# remove HTML tags, convert to lowercase, remove stopwords, punctuation and special characters
def preprocess_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df['Text'] = df['Text'].apply(preprocess_text)

positive_reviews = df[df['Label'] == 1]['Text']
negative_reviews = df[df['Label'] == 0]['Text']

# generate wordcloud
def generate_wordcloud(text_series, title):
    text = ' '.join(text_series)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(title + '.png')
    plt.show()

generate_wordcloud(positive_reviews, 'Positive Reviews')

generate_wordcloud(negative_reviews, 'Negative Reviews')

df.to_csv('balanced_preprocess.csv', index=False)

####################tfidf_nn code####################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# check gpu availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# load the balanced dataset
# balanced_sample_df = pd.read_csv('balanced_reviews_sample.csv')  
# balanced_sample_df = pd.read_csv('balanced_preprocess.csv')
balanced_sample_df = pd.read_csv('preprocess_simple.csv')
X = balanced_sample_df['Text']
y = balanced_sample_df['Label']

# TFIDF vectorization
tfidf = TfidfVectorizer(max_features=10000) 
X_tfidf = tfidf.fit_transform(X).toarray()

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# transform the data into tensors
X_train_tensor = torch.FloatTensor(X_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_train_tensor = torch.LongTensor(y_train.values).to(device)
y_test_tensor = torch.LongTensor(y_test.values).to(device)

# create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# define the model
class TextClassifier(nn.Module):
    def __init__(self, input_dim):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x) 
        x = self.fc2(x)
        x = self.softmax(x)
        return x

model = TextClassifier(input_dim=X_train.shape[1]).to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train the model
num_epochs = 10
for epoch in range(num_epochs):
    for texts, labels in train_loader:
        # forward pass
        outputs = model(texts)
        loss = criterion(outputs, labels)
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# use the model to make predictions
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)

precision = precision_score(y_test_tensor.cpu(), predicted.cpu())
recall = recall_score(y_test_tensor.cpu(), predicted.cpu())

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1_score(y_test_tensor.cpu(), predicted.cpu())}')

cm = confusion_matrix(y_test_tensor.cpu(), predicted.cpu())
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_tfidf.png')
plt.show()

####################tfidf_lr code####################

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# load the balanced dataset
# balanced_sample_df = pd.read_csv('balanced_reviews_sample.csv')  
# balanced_sample_df = pd.read_csv('balanced_preprocess.csv')
balanced_sample_df = pd.read_csv('preprocess_simple.csv')
X = balanced_sample_df['Text']
y = balanced_sample_df['Label']

# TFIDF vectorization
tfidf = TfidfVectorizer(max_features=10000) 
X_tfidf = tfidf.fit_transform(X).toarray()

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# define the model
model = LogisticRegression(max_iter=1000)

# train the model
model.fit(X_train, y_train)

# use the model to make predictions
predicted = model.predict(X_test)
accuracy = (predicted == y_test).sum() / len(y_test)

precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1_score(y_test, predicted)}')

cm = confusion_matrix(y_test, predicted)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
plt.savefig('confusion_matrix_lr_tfidf.png')

####################word2vec_nn code####################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import gensim.downloader as gensim_api
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load pre-trained Word2Vec model
word2vec_model = gensim_api.load("word2vec-google-news-300")

# Define a function to get the vector for a document
def document_vector(doc):
    # remove out-of-vocabulary words
    words = doc.split()
    word_vectors = [word2vec_model[word] for word in words if word in word2vec_model]
    if not word_vectors:  # if the list is empty
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word_vectors, axis=0)

# Load the balanced dataset
# balanced_sample_df = pd.read_csv('balanced_reviews_sample.csv')  
# balanced_sample_df = pd.read_csv('balanced_preprocess.csv')  
balanced_sample_df = pd.read_csv('preprocess_simple.csv')
X = balanced_sample_df['Text']
y = balanced_sample_df['Label']

# Obtain Word2Vec vectors for the text data
X_word2vec = np.array([document_vector(text) for text in X])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_word2vec, y, test_size=0.2, random_state=42)

# Transform the data into tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

batch_size = 32

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual  
        out = self.relu(out)
        return out

class DeepTextClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, block, layers):
        super(DeepTextClassifier, self).__init__()
        self.input_fc = nn.Linear(input_dim, 512)
        self.bn_input = nn.BatchNorm1d(512)
        self.relu = nn.ReLU(inplace=True)
        self.layers = self._make_layer(block, 512, layers)
        self.output_fc = nn.Linear(512, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def _make_layer(self, block, input_dim, layers):
        layers_list = []
        for _ in range(layers):
            layers_list.append(block(input_dim))
        return nn.Sequential(*layers_list)
    
    def forward(self, x):
        x = self.input_fc(x)
        x = self.bn_input(x)
        x = self.relu(x)
        x = self.layers(x)
        x = self.output_fc(x)
        x = self.log_softmax(x)
        return x

class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(feature_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, num_features, feature_dim)
        weights = torch.softmax(self.attention(x), dim=1)
        # Apply attention weights
        weighted = torch.mul(x, weights)
        # Sum over the feature dimension
        attended = weighted.sum(1)
        return attended
    
num_classes = 2  
num_layers = 4  
model = DeepTextClassifier(input_dim=300, num_classes=num_classes, block=ResidualBlock, layers=num_layers).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3, verbose=True)

# Train the model
num_epochs = 50

for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Validation phase
    model.eval()  
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for texts, labels in val_loader: 
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    
    accuracy = correct_predictions / total_predictions
    
    # Print metrics
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}, Accuracy: {accuracy}')
    
    # Step the scheduler
    scheduler.step(val_loss/len(val_loader))

all_predicted = []
all_labels = []

with torch.no_grad():
    for texts, labels in val_loader:
        texts = texts.to(device)
        outputs = model(texts)
        _, predicted = torch.max(outputs, 1)
        all_predicted.extend(predicted.cpu().numpy()) 
        all_labels.extend(labels.cpu().numpy())  

precision = precision_score(all_labels, all_predicted)
recall = recall_score(all_labels, all_predicted)
accuracy = (np.array(all_predicted) == np.array(all_labels)).mean()

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

# Confusion Matrix
cm = confusion_matrix(all_labels, all_predicted)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_word2vec_re.png')
plt.show()

####################BERT_without code####################

from transformers import pipeline
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the balanced dataset
file_path = 'balanced_reviews_sample.csv'  
balanced_sample_df = pd.read_csv(file_path)

# Select texts for sentiment analysis
texts = balanced_sample_df['Text'].tolist()

# Initialize the sentiment-analysis pipeline with BERT
classifier = pipeline("sentiment-analysis", device=0)  

# Perform sentiment analysis
results = []
for text in texts:
    max_length = 512
    truncated_text = text[:max_length]
    result = classifier(truncated_text)
    results.append(result[0]['label']) 

# Append the results back to the DataFrame
balanced_sample_df['Predicted_Sentiment'] = results

# Convert sentiment labels to binary (0/1) for metrics calculation
balanced_sample_df['Predicted_Sentiment'] = balanced_sample_df['Predicted_Sentiment'].apply(lambda x: 1 if x == 'POSITIVE' else 0)

# Calculate metrics
accuracy = accuracy_score(balanced_sample_df['Label'], balanced_sample_df['Predicted_Sentiment'])
precision = precision_score(balanced_sample_df['Label'], balanced_sample_df['Predicted_Sentiment'])
recall = recall_score(balanced_sample_df['Label'], balanced_sample_df['Predicted_Sentiment'])
f1 = f1_score(balanced_sample_df['Label'], balanced_sample_df['Predicted_Sentiment'])

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Display the first few sentiment analysis results
print(balanced_sample_df[['Text', 'Predicted_Sentiment']].head())

####################BERT_with code####################

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

####################BERT_withLora code####################

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
    evaluation_strategy='steps',  
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
# train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)  

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
