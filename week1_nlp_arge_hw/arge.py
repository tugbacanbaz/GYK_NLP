import zipfile
import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import NMF
import numpy as np
import re


zip_path = 'news-category-dataset.zip'
extract_path = 'data'
json_filename = 'News_Category_Dataset_v3.json'
json_path = os.path.join(extract_path, json_filename)

if not os.path.exists(extract_path):
    os.makedirs(extract_path)

if not os.path.exists(json_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Dataset extracted.")
else:
    print("Dataset already extracted.")


#Load the dataset
df = pd.read_json(json_path, lines=True)

#df structure
print(df.head())
print("df column names: ", df.columns)
print("Column data types: ", df.dtypes)


# Downloading requirements
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def preprocess_headlines(corpus):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    clean_text = []

    for text in corpus:
        #Lowercasing
        text =text.lower()
        # Replace dashes BEFORE tokenizing
        text = text.replace("-", " ")
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        #Tokenization
        tokens = word_tokenize(text)
        #Stopword removal
        #To remove unwanted tokens like punctuation, numbers and mixed strings used word.isalpha()
        filtered_tokens =[word for word in tokens if tokens and word not in stop_words]
        #POS tagging
        pos_tags = pos_tag(filtered_tokens)
        #Lemmatization
        lemmatized_tokens = []
        for word, tag in pos_tags:
            if tag.startswith('J'):
                pos = 'a'
            elif tag.startswith('V'):
                pos = 'v'
            elif tag.startswith('N'):
                pos = 'n'
            elif tag.startswith('R'):
                pos = 'r'
            else:
                pos = 'n'

            lemmanized = lemmatizer.lemmatize(word, pos=pos)
            lemmatized_tokens.append(lemmanized)
        clean_text.append(" ".join(lemmatized_tokens))
    return clean_text

#testing preprocessing
sample_headlines = df['headline'].head(20)
sample_cleaned = preprocess_headlines(sample_headlines)

for original, processed in zip(sample_headlines, sample_cleaned):
    print("Original Sample  :", original)
    print("Cleaned Sample :", processed, "\n")

#since it works lets work 1000 datapoints insteaad of entire dataset for memory
headlines = df['headline'].head(1000)
cleaned = preprocess_headlines(headlines)

# CountVectorizer (Bag of Words)
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(cleaned)

print("CountVectorizer Feature Names:")
print(count_vectorizer.get_feature_names_out()[:20]) 

print("CountVectorizer Count Matrix (first 5 rows): ")
print(count_matrix.toarray()[:5])  

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned)

print("\n TF-IDF - Feature Names:")
print(tfidf_vectorizer.get_feature_names_out()[:20])

print("TF-IDF - TF-IDF Matrix (first 5 rows):")
print(tfidf_matrix.toarray()[:5])

#Her yöntemin avantaj/dezavantajlarını örnek çıktılarla gösterin.
# CountVectorizer (Bag of Words)
row = count_matrix[0]
words = count_vectorizer.get_feature_names_out()
print([(words[i], row[0, i]) for i in row.nonzero()[1]])
# TF-IDF Vectorizer
row_tfidf = tfidf_matrix[0]
words_tfidf = tfidf_vectorizer.get_feature_names_out()
print([(words_tfidf[i], row_tfidf[0, i]) for i in row.nonzero()[1]])

#We applied both CountVectorizer and TF-IDF Vectorizer to the cleaned headlines
#The vectorizers analyzed the entire set of 1000 headlines and built a vocabulary from all unique tokens
#As seen in the first row of the matrix:
#CountVectorizer gives raw frequency counts (e.g., ('covid': 1))
#TF-IDF gives importance scores (e.g., ('covid': 0.42)), which are influenced by how often that word appears across all documents

#Sentiment Analysis with TextBlob library
nltk.download('brown')
nltk.download('punkt') 

def get_sentiment_label(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # between -1 and +1
    if polarity > 0.05:
        return "Positive"
    elif polarity < -0.05:
        return "Negative"
    else:
        return "Neutral"

#We performed sentiment analysis on the first 20 English news headlines using the TextBlob library 
#The headlines were first preprocessed (tokenized, stopword-removed, lemmatized), and the cleaned text was passed into TextBlob to extract a polarity score
#which was mapped to one of three sentiment classes: Positive: polarity > 0.05, Negative: polarity < -0.05, Neutral: -0.05 ≤ polarity ≤ 0.05


#Sentiment Analysis with VADER (Valence Aware Dictionary for sEntiment Reasoning)

vader_analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment_label(text):
    scores = vader_analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"
    
#Also used VADER library
    
print("Comparing TextBlob and VADER on first 20 headlines:\n")
for original, cleaned_text in zip(headlines[:20], cleaned[:20]):
    textblob_sent = get_sentiment_label(cleaned_text)
    vader_sent = get_vader_sentiment_label(cleaned_text)
    
    print(f"Original Headline: {original}")
    print(f"Cleaned  Headline: {cleaned_text}")
    print(f"TextBlob Sentiment: {textblob_sent}")
    print(f"VADER Sentiment   : {vader_sent}")
    print("\n")


#Both models generally agree on highly polarized cases — especially when the sentiment is clearly Positive (e.g., headlines about something "adorable" or "funny") 
#or clearly Negative (e.g., death, violence).
#However, VADER tends to assign more polar sentiment (Positive or Negative) compared to TextBlob, which is more conservative and leans toward Neutral.
#TextBlob Positive 6, Negative 4 Neutral 10
#VADER Positive 7 Negative 7 Neutral 6


#Topic Modeling

# NMF
num_topics = 5

# Fit NMF model
nmf_model = NMF(n_components=num_topics, random_state=42)
nmf_model.fit(tfidf_matrix)

feature_names = tfidf_vectorizer.get_feature_names_out()

def display_topics(model, feature_names, no_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        print(f"\n Topic {topic_idx + 1}:")
        print("Top Words:", ", ".join(top_words))

display_topics(nmf_model, feature_names)

#Topic 1: Daily News
#Topic 2: Politics
#Topic 3: War
#Topic 4: Law
#Topic 5: Emergency