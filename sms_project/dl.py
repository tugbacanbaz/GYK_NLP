import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


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

#Tokenizer
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
print(tokenizer.word_index.items())

#sequences
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

#original texts
for i in range(3):
    print(f"{i+1}. Text: {X_train.iloc[i]}")
#sequences
for i in range(3):
    print(f"{i+1}. Sequence: {X_train_sequences[i]}")

#padding
max_length = 100
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding='post')

for i in range(3):
    print(f"{i+1}. Text: {X_train_padded[i]}")

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=16, input_length=max_length))

#Long Short-Term Memory (LSTM) layer
#RNN (Recurrent Neural Network) layer
model.add(LSTM(64))
#Dense fully connected layer
model.add(Dense(1, activation='sigmoid')) #sigmoid -> 0-1

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
history = model.fit(X_train_padded, y_train, epochs=10, validation_data=(X_test_padded, y_test))
