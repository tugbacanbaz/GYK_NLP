import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1','v2']]
df.columns =['label', 'text']

df['label'] = df['label'].map({'ham':0, 'spam':1})

print(df.head())

#Normalize the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+','',text) #remove numbers
    text = text.translate(str.maketrans('','', string.punctuation) ) #remove punctuation
    return text

df['clean_text'] = df['text'].apply(clean_text)
print(df.head())

#train test split
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'],df['label'], test_size=0.2, random_state=42)

#Vectorizer
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

#numeric dictionary
print(vectorizer.get_feature_names_out())
print(X_train_vectorized.toarray())

#naive bayes
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

#Benchmark
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#new_sms = "I'm sorry to hear that you're having trouble with your account. Let me know if I can help you writ anything."
new_sms = " You have won a free iPhone 13 Pro Max! Click the link to claim your prize. "

new_sms_cleaned = clean_text(new_sms)
new_sms_vectorized = vectorizer.transform([new_sms_cleaned])

prediction = model.predict(new_sms_vectorized)

print("Prediction for new SMS: ", "Spam" if prediction[0] == 1 else "Ham")