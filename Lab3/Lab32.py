import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

wowie = pd.read_csv('C:/Users/nikit/Documents/GitHub/TNM098/Lab3/Lab3.2/TNM098-MC3-2011.csv', sep=';')

# Renmove punctuation, convert to lowercase remove stop words
wowie['Content'] = wowie['Content'].str.replace(r'[^\w\s]', '', regex=True).str.lower()
wowie['Content'] = wowie['Content'].str.split()
wowie['Content'] = wowie['Content'].apply(lambda x: [word for word in x if word not in ['and', 'the', 'is', 'to', 'a', 'in', 'of', 'for', 'on', 'that', 'this', 'it', 'with']])

# Tokenize the text
wowie['Content'] = wowie['Content'].apply(lambda x: ' '.join(x))

# Histogram of the temporal data
plt.figure(figsize=(10, 6))
plt.hist(wowie['Date'], bins=30, color='blue', alpha=0.7)
plt.title('Histogram of Temporal Data')
plt.xlabel('Date')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


