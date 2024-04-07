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