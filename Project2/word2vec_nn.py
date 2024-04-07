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
