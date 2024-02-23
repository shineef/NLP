import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
import time
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize lists to store the true labels and predicted labels
true_labels = []
predicted_labels = []

# Load the datasets
true_df = pd.read_csv('true.csv')
fake_df = pd.read_csv('fake.csv')

# Assign labels
true_df['label'] = 1
fake_df['label'] = 0

# Combine the datasets
df = pd.concat([true_df, fake_df]).reset_index(drop=True)

# Preprocess the text data
# nltk.download('punkt')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

def get_pos_tags(text, desired_pos=['NN', 'JJ']):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    filtered_words = [word for word, tag in tagged if tag in desired_pos]
    return filtered_words

df['pos_tags'] = df['text'].apply(get_pos_tags)

df['text'] = df['text'].apply(preprocess_text)
df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

mlb = MultiLabelBinarizer()
pos_encoded = mlb.fit_transform(df['pos_tags'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Add the sentiment score as a feature
X_train_sentiment = df.loc[X_train.index, 'sentiment'].values.reshape(-1, 1)
X_test_sentiment = df.loc[X_test.index, 'sentiment'].values.reshape(-1, 1)

X_train_combined = np.hstack((X_train_tfidf, X_train_sentiment, pos_encoded[:len(X_train)]))
X_test_combined = np.hstack((X_test_tfidf, X_test_sentiment, pos_encoded[len(X_train):]))

# Convert to PyTorch tensors
train_features = torch.tensor(X_train_combined, dtype=torch.float32)
test_features = torch.tensor(X_test_combined, dtype=torch.float32)
train_labels = torch.tensor(y_train.values, dtype=torch.long)
test_labels = torch.tensor(y_test.values, dtype=torch.long)

train_features = train_features.reshape(train_features.shape[0], 1, -1)  
test_features = test_features.reshape(test_features.shape[0], 1, -1)  

# Create a Dataset
class NewsDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Create DataLoader
batch_size = 32
train_dataset = NewsDataset(train_features, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = NewsDataset(test_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# CNN Model
class TextCNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(TextCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=4, padding=2)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(5000)  
        self.fc = nn.Linear(300 * 5000, num_classes)  
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(x))
        x3 = torch.relu(self.conv3(x))

        # Adaptive pooling
        x1 = self.adaptive_pool(x1)
        x2 = self.adaptive_pool(x2)
        x3 = self.adaptive_pool(x3)

        # Concatenate along the feature dimension
        x = torch.cat((x1, x2, x3), 1)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Initialize the CNN model
num_features = train_features.shape[1]
num_classes = 2
model = TextCNN(num_features, num_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device) 
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.update(1)  # Update the progress bar

        pbar.set_postfix({'Loss': total_loss / len(train_loader)})

# Evaluate the model
model.eval() 
with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        outputs = model(batch_features)
        _, predicted = torch.max(outputs.data, 1)
        # Append batch prediction results
        true_labels.extend(batch_labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

# Calculate precision, recall, and confusion matrix
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels)
accuracy = accuracy_score(true_labels, predicted_labels)

end_time = time.time()

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print(f"Execution time: {end_time - start_time} seconds")