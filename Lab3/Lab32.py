import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA


# Relative path to the CSV file
wowie = pd.read_csv(r'Lab3\Lab3.2\TNM098-MC3-2011.csv', sep=';')

# Renmove punctuation, convert to lowercase remove stop words
wowie['Content'] = wowie['Content'].str.replace(r'[^\w\s]', '', regex=True).str.lower()
wowie['Content'] = wowie['Content'].str.split()

# Tokenize the text using NLTK and remove stop words
#nltk.download('stopwords')  # Uncomment this line if stopwords are not downloaded
stop_words = set(stopwords.words('english'))
wowie['Content'] = wowie['Content'].apply(lambda x: ' '.join([word for word in x if word not in stop_words]))

# Sort the data by date
wowie['Date'] = pd.to_datetime(wowie['Date'], format='%Y-%m-%d')
wowie = wowie.sort_values(by='Date')

#Filter the data to only include relevant keywords
keywords = ["threat", "terrorism", "terrorist", "attack", "bomb", "explosion", "shooting", "violence", "dead", "injured", 
            "casualties", "victims", "injury", "wounded", "assault", "hostage", "kidnapping", "explosive", "gunfire", "gunman", "militant"]

# First create a copy of the dataframe
filtered_wowie = wowie.copy()

# Create a mask that identifies rows containing at least one keyword
mask = filtered_wowie['Content'].apply(lambda x: any(keyword in x.split() for keyword in keywords))

# Apply the mask to keep only rows with at least one keyword
filtered_wowie = filtered_wowie[mask]

# Then still filter the content to only contain keywords
filtered_wowie['Content'] = filtered_wowie['Content'].apply(lambda x: ' '.join([word for word in x.split() if word in keywords]))


# Histogram of the temporal data
plt.figure(figsize=(10, 6))
# Plot with the same bins and no normalization
plt.hist(wowie['Date'], bins=40, color='blue', alpha=0.7)
plt.hist(filtered_wowie['Date'], bins=40, color='red', alpha=0.7)
plt.legend(['Original Data', 'Filtered Data'])
plt.title('Histogram of Temporal Data')
plt.xlabel('Date')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(wowie['Content'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

#print(tfidf_df.head())
#print(wowie['Content'].head())

# Plotting the TF-IDF matrix
'''plt.figure(figsize=(10, 6))
plt.imshow(tfidf_matrix.toarray(), cmap='hot', interpolation='nearest')
plt.title('TF-IDF Matrix Heatmap')
plt.colorbar(label='TF-IDF Value')
plt.xlabel('Terms')
plt.ylabel('Documents')
plt.xticks(ticks=np.arange(len(vectorizer.get_feature_names_out())), labels=vectorizer.get_feature_names_out(), rotation=90)
plt.yticks(ticks=np.arange(len(wowie)), labels=wowie['Content'].index)
plt.tight_layout()
plt.show()'''

# Select top n terms based on their average TF-IDF scores
n_terms = 50  # Number of terms to display
n_docs = 20   # Number of documents to display (Max 58)

# Calculate the average TF-IDF score for each term
term_scores = tfidf_matrix.mean(axis=0).A1
top_term_indices = np.argsort(term_scores)[-n_terms:]  # Indices of top n terms

# Select a subset of documents
subset_docs = wowie['Content'].index[:n_docs]

# Create a smaller TF-IDF matrix for plotting
reduced_tfidf_matrix = tfidf_matrix[subset_docs, :][:, top_term_indices]
reduced_terms = vectorizer.get_feature_names_out()[top_term_indices]

# Plotting the reduced TF-IDF matrix
plt.figure(figsize=(10, 6))
plt.imshow(reduced_tfidf_matrix.toarray(), cmap='hot', interpolation='nearest')
plt.title('Reduced TF-IDF Matrix Heatmap')
plt.colorbar(label='TF-IDF Value')
plt.xlabel('Top Terms')
plt.ylabel('Documents')
plt.xticks(ticks=np.arange(len(reduced_terms)), labels=reduced_terms, rotation=90, fontsize=6)
plt.yticks(ticks=np.arange(len(subset_docs)), labels=subset_docs, fontsize=6)
plt.tight_layout()
plt.show()

# Build topic model (LDA)
n_topics = 5  # Number of topics
lda = LDA(n_components=n_topics, random_state=42)
lda.fit(tfidf_matrix)

# Display topic related words
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx}:")
    top_feature_indices = topic.argsort()[-10:][::-1]  # Top 10 words for each topic
    top_features = vectorizer.get_feature_names_out()[top_feature_indices]
    print(" ".join(top_features))
    print()

# Generate topic labels based on the top terms for each topic
topic_labels = []
for topic_idx, topic in enumerate(lda.components_):
    top_feature_indices = topic.argsort()[-5:][::-1]  # Top 5 words for each topic
    top_features = vectorizer.get_feature_names_out()[top_feature_indices]
    topic_labels.append(", ".join(top_features))  # Combine top terms into a single label

# Console log the topic labels
print("\nTopic Labels:")
for idx, label in enumerate(topic_labels):
    print(f"Topic {idx}: {label}")

