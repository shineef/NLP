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