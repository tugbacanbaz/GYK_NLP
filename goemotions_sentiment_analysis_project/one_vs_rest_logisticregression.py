import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle 
from sklearn.multiclass import OneVsRestClassifier

# Load the dataset
df = pd.read_csv("data/dataset/goemotions_cleaned.csv")

df = df.dropna(subset=['clean_text'])  # Drop rows with NaN in 'clean_text'
X = df['clean_text'].values

emotion_columns = df.columns[9:37]
df[emotion_columns] = df[emotion_columns].astype(int)
y = df[emotion_columns].values

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=df.columns[9:37]))
print("Accuracy:", accuracy_score(y_test, y_pred))


