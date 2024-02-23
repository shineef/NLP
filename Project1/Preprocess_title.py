import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from wordcloud import WordCloud

# Load the datasets
true_news_df = pd.read_csv('true.csv')
fake_news_df = pd.read_csv('fake.csv')

# Define preprocessing function
def preprocess_title(title):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(title)
    lemmatizer = WordNetLemmatizer()
    # Lemmatize and remove stop words and non-alphabetic characters
    lemmatized_output = [lemmatizer.lemmatize(w) for w in word_tokens if not w.lower() in stop_words and w.isalpha()]
    return lemmatized_output

# Preprocess the titles
true_news_df['processed_title'] = true_news_df['title'].apply(preprocess_title)
fake_news_df['processed_title'] = fake_news_df['title'].apply(preprocess_title)

# Concatenate all the processed words from titles
all_true_titles = sum(true_news_df['processed_title'].tolist(), [])
all_fake_titles = sum(fake_news_df['processed_title'].tolist(), [])

# Get the frequency distribution of the words in titles
true_title_freq = nltk.FreqDist(all_true_titles)
fake_title_freq = nltk.FreqDist(all_fake_titles)

# Get the top 100 words in titles
top100_true_titles = true_title_freq.most_common(100)
top100_fake_titles = fake_title_freq.most_common(100)

# Create DataFrame from the top 100 words in titles
top_titles_df = pd.DataFrame({
    'Real_News_Title_Word': [word for word, freq in top100_true_titles],
    'Real_News_Title_Frequency': [freq for word, freq in top100_true_titles],
    'Fake_News_Title_Word': [word for word, freq in top100_fake_titles],
    'Fake_News_Title_Frequency': [freq for word, freq in top100_fake_titles],
})

# Save to CSV
top_titles_df.to_csv('TOP100title.csv', index=False)

# Function to generate word cloud images for titles
def generate_title_word_cloud(frequencies, filename):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(frequencies))
    wordcloud.to_file(filename + '.png')

# Generate word clouds for both real and fake news
generate_title_word_cloud(top100_true_titles, 'real_news_title_wordcloud')
generate_title_word_cloud(top100_fake_titles, 'fake_news_title_wordcloud')