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