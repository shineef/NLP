import pandas as pd
import matplotlib.pyplot as plt

# Load the Data
file_path = 'Reviews.csv'
reviews_df = pd.read_csv(file_path)

# Since we only use scores and text, lets see score distribution for this part
score_distribution = reviews_df['Score'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
score_distribution.plot(kind='bar')
plt.title('Distribution of Review Scores')
plt.xlabel('Score')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.savefig('score_distribution.png')
plt.show()

# Label Creation
reviews_filtered_df = reviews_df[reviews_df['Score'] != 3]

# Create a new column for labels: 1 for positive (score > 3) and 0 for negative (score < 3)
reviews_filtered_df['Label'] = (reviews_filtered_df['Score'] > 3).astype(int)

# Check the balance of the newly created labels
label_distribution = reviews_filtered_df['Label'].value_counts(normalize=True)
print(label_distribution)

# Separate positive and negative reviews
positive_reviews = reviews_filtered_df[reviews_filtered_df['Label'] == 1]
negative_reviews = reviews_filtered_df[reviews_filtered_df['Label'] == 0]

# determine the sample size
total_count = len(reviews_filtered_df)
target_sample_size = int(total_count * 0.2)  
negative_count = len(negative_reviews)

if negative_count >= target_sample_size / 2:
    positive_sample = positive_reviews.sample(n=target_sample_size // 2, random_state=42)
    negative_sample = negative_reviews.sample(n=target_sample_size // 2, random_state=42)
else:
    negative_sample = negative_reviews
    positive_sample = positive_reviews.sample(n=negative_count, random_state=42)

# combine the samples to form a balanced dataset
balanced_samples = pd.concat([positive_sample, negative_sample])

total_balanced_samples = balanced_samples.shape[0]
total_original_samples = reviews_df.shape[0]
proportion_balanced_to_original = total_balanced_samples / total_original_samples

save_path = 'balanced_reviews_sample.csv'
balanced_sample_file_path = balanced_samples.to_csv(save_path, index=False)

# Print statements to see the values (you can comment these out if needed)
print(f"Total samples in balanced dataset: {total_balanced_samples}")
print(f"Proportion of balanced dataset to original: {proportion_balanced_to_original}")
print(f"Balanced dataset saved to: {balanced_sample_file_path}")
