import nltk
import re
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer


#Load and define corpus
corpus = [
    "Artificial Intelligence is the future.",
    "AI is changing the world.",
    "AI is a branch of computer science.",
]

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Fonkisyon
# pipeline => 
# 1-Tokenization - lowercasing 
# 2- Stopwords Temizliği
# 3- Lemmatization
# 4- TF-IDF Vektörleştirme
# 5- Feature isimlerini ve arrayi ekrana yazdır.

def nlp_func(corpus):
    cleaned_corpus = []
    for doc in corpus:
        # 1-Tokenization - lowercasing 
        doc_lower = doc.lower()
        tokens = word_tokenize(doc_lower)
        print("Tokens: ", tokens)

        # 2- Stopwords Temizliği
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        print("Tokens without stopwords: ", filtered_tokens)

        # 3- Lemmatization
        lemmatizer = WordNetLemmatizer()
        pos_tags = pos_tag(filtered_tokens)
        print("Pos tags: ", pos_tags)
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

            lemma = lemmatizer.lemmatize(word, pos=pos)
            lemmatized_tokens.append(lemma)

        print("After lemmatization:", lemmatized_tokens)
        print("\n")

        cleaned_sentence = " ".join(lemmatized_tokens)
        cleaned_corpus.append(cleaned_sentence)

    return cleaned_corpus

processed_corpus = nlp_func(corpus)
# 4- TF-IDF Vektörleştirme
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_corpus)

# 5- Feature isimlerini ve arrayi ekrana yazdır.

print("Get feature names: ", vectorizer.get_feature_names_out())
print("TF-IDF Array: ", X.toarray())