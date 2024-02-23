import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from wordcloud import WordCloud

# Ensure the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the datasets
true_news_df = pd.read_csv('true.csv')
fake_news_df = pd.read_csv('fake.csv')

# Define preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    # Lemmatize and remove stop words and non-alphabetic characters
    lemmatized_output = [lemmatizer.lemmatize(w) for w in word_tokens if not w.lower() in stop_words and w.isalpha()]
    return lemmatized_output

# Preprocess the texts
true_news_df['processed'] = true_news_df['text'].apply(preprocess_text)
fake_news_df['processed'] = fake_news_df['text'].apply(preprocess_text)

# Concatenate all the processed words
all_true_words = sum(true_news_df['processed'].tolist(), [])
all_fake_words = sum(fake_news_df['processed'].tolist(), [])

# Get the frequency distribution of the words
true_word_freq = nltk.FreqDist(all_true_words)
fake_word_freq = nltk.FreqDist(all_fake_words)

# Get the top 100 words
top100_true_words = true_word_freq.most_common(100)
top100_fake_words = fake_word_freq.most_common(100)

# Create DataFrame from the top 100 words
top_words_df = pd.DataFrame({
    'Real_News_Word': [word for word, freq in top100_true_words],
    'Real_News_Frequency': [freq for word, freq in top100_true_words],
    'Fake_News_Word': [word for word, freq in top100_fake_words],
    'Fake_News_Frequency': [freq for word, freq in top100_fake_words],
})

# Save to CSV
top_words_df.to_csv('TOP100text.csv', index=False)

# Function to generate word cloud images
def generate_word_cloud(frequencies, filename):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(frequencies))
    wordcloud.to_file(filename + '.png')

# Generate word clouds for both real and fake news
generate_word_cloud(top100_true_words, 'real_news_wordcloud')
generate_word_cloud(top100_fake_words, 'fake_news_wordcloud')
