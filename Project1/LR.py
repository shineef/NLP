import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time

# nltk.download('vader_lexicon')
# nltk.download('averaged_perceptron_tagger')

start_time = time.time()

# Load the datasets
true_df = pd.read_csv('true.csv')
fake_df = pd.read_csv('fake.csv')

# Label the datasets
true_df['label'] = 1
fake_df['label'] = 0

# Combine the datasets
df = pd.concat([true_df, fake_df], ignore_index=True)

# Preprocessing function for sentiment scores
def get_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)['compound']

# Preprocessing function for part-of-speech tags
def get_pos_tags(text):
    pos_tags = nltk.pos_tag(nltk.word_tokenize(text))
    pos_counts = {'noun_count': 0, 'verb_count': 0, 'adj_count': 0, 'adv_count': 0}
    for _, tag in pos_tags:
        if tag.startswith('N'): pos_counts['noun_count'] += 1
        elif tag.startswith('V'): pos_counts['verb_count'] += 1
        elif tag.startswith('J'): pos_counts['adj_count'] += 1
        elif tag.startswith('R'): pos_counts['adv_count'] += 1
    return pos_counts

# Apply get_pos_tags to the 'text' column and expand the dictionary into separate columns
pos_tags_df = df['text'].apply(get_pos_tags).apply(pd.Series)

# Concatenate the new POS tag columns to the original DataFrame
df = pd.concat([df, pos_tags_df], axis=1)

# print(df.columns)

# Add sentiment scores and POS tag counts as new columns
df['sentiment'] = df['text'].apply(get_sentiment)
df['pos_tags'] = df['text'].apply(get_pos_tags)

X = df[['text', 'sentiment', 'noun_count', 'verb_count', 'adj_count', 'adv_count']]  
y = df['label']

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df[['text', 'sentiment', 'pos_tags']], df['label'], test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define feature extraction methods for different features
feature_transformer = ColumnTransformer(
    transformers=[
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2)), 'text'),
        ('sentiment', 'passthrough', ['sentiment']),
        ('noun_count', 'passthrough', ['noun_count']),
        ('verb_count', 'passthrough', ['verb_count']),
        ('adj_count', 'passthrough', ['adj_count']),
        ('adv_count', 'passthrough', ['adv_count'])
    ],
    remainder='drop'
)

# Define the Logistic Regression pipeline
pipeline = Pipeline(steps=[
    ('features', feature_transformer),
    ('scaler', StandardScaler(with_mean=False)),  
    ('classifier', LogisticRegression(max_iter=1000, n_jobs=20))  
])

# Fit the model
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

end_time = time.time()

# Calculate performance metrics
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# print('Accuracy:', accuracy)
print(f"Execution time: {end_time - start_time} seconds")