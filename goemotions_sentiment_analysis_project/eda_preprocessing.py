import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string


df = pd.read_csv("data/dataset/goemotions_merged.csv")

#eda
print(df.head())
print(df.info())

#emotion distribution
emotion_columns = df.columns[9:]


emotion_counts = df[emotion_columns].sum().sort_values(ascending=False)

plt.figure(figsize=(14, 6))
emotion_counts.plot(kind='bar')
plt.title("Emotion Frequencies in Dataset")
plt.ylabel("Count")
plt.xlabel("Emotion")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#correlation matrix
plt.figure(figsize=(18, 16))
corr = df[emotion_columns].corr()

sns.heatmap(
    corr,
    cmap='coolwarm',
    annot=True,
    fmt=".2f",
    annot_kws={"size": 8},  
    square=True,
    cbar=True
)

plt.title("Emotion Co-occurrence Correlation Matrix", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#preprocessing

def clean_text(text):
    text = text.lower()  #small case
    text = re.sub(r"[^\x00-\x7F]+", "", text)  #remove non-ASCII characters
    text = re.sub(r"http\S+", "", text)  #remove URL
    text = re.sub(r"\[.*?\]", "", text)  #remove [NAME], [RELIGION]
    text = re.sub(r"@\w+", "", text)  #remove mention 
    text = re.sub(r"#\w+", "", text)  #remove hashtag 
    text = text.translate(str.maketrans("", "", string.punctuation))  #remove punctuation
    text = re.sub(r"\d+", "", text)  #remove numbers
    text = re.sub(r"\s+", " ", text).strip()  #remove extra spaces
    return text


df['clean_text'] = df['text'].apply(clean_text)
print(df[['text', 'clean_text']].sample(5))

#Save the cleaned DataFrame
df.to_csv("data/dataset/goemotions_cleaned.csv", index=False)