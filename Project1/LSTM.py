import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
import time
from nltk import pos_tag

start_time = time.time()

# nltk.download('punkt')

# Load dataset
true_df = pd.read_csv('true.csv')
fake_df = pd.read_csv('fake.csv')

true_df['label'] = 1
fake_df['label'] = 0
df = pd.concat([true_df, fake_df])

def filter_pos(text, desired_pos = ['NN', 'JJ']):
    tokens = word_tokenize(text.lower())
    tagged_tokens = pos_tag(tokens)
    filtered_tokens = [word for word, tag in tagged_tokens if tag in desired_pos]
    return ' '.join(filtered_tokens)

df['text'] = df['text'].apply(lambda text: filter_pos(text, desired_pos= ['VB']))

# Tokenization and building vocabulary
def tokenize(text):
    tokens = word_tokenize(text.lower())
    return tokens


def build_vocab(texts):
    token_freqs = Counter()
    for text in texts:
        tokens = tokenize(text)
        token_freqs.update(tokens)
    vocab = {token: idx + 2 for idx, (token, _) in enumerate(token_freqs.items())}
    vocab['<PAD>'] = 0  # Padding token
    vocab['<UNK>'] = 1  # Unknown token
    return vocab


vocab = build_vocab(df['text'])


# Numericalize text
def numericalize(text, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in tokenize(text)]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)


# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = [torch.tensor(numericalize(text, vocab), dtype=torch.long) for text in texts]
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# Padding sequences for batching
def collate_fn(batch):
    texts, labels = zip(*batch)
    max_length = max(len(text) for text in texts)
    padded_texts = torch.zeros(len(texts), max_length, dtype=torch.long)
    for i, text in enumerate(texts):
        padded_texts[i, :len(text)] = text
    return padded_texts, torch.tensor(labels)


# Create DataLoader
train_dataset = TextDataset(X_train, y_train, vocab)
test_dataset = TextDataset(X_test, y_test, vocab)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


class LSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, _) = self.lstm(embedded)
        hidden = hidden.squeeze(0)
        output = self.fc(hidden)
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMTextClassifier(len(vocab), 100, 256, 2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(8):  # Training loop
    model.train()
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

model.eval()
true_labels = []
predicted_labels = []

with torch.no_grad():
    for texts, labels in test_loader:
        texts, labels = texts.to(device), labels.to(device)
        outputs = model(texts)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
accuracy = accuracy_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels)

end_time = time.time()

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print(f"Execution time: {end_time - start_time} seconds")