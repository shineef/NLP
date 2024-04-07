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
