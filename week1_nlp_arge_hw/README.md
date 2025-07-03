# NLP Ar-Ge Project: Sentiment Analysis & Topic Modeling on News Headlines

This project focuses on applying core **Natural Language Processing (NLP)** techniques to analyze news headlines. We aim to extract meaningful insights by performing **sentiment analysis** and **topic modeling** using both classical and transformer-based methods.

---

## Objective

To develop hands-on experience with essential NLP workflows by:

- Preprocessing and cleaning raw text data
- Converting text into numerical representations
- Performing sentiment analysis on headlines
- Identifying latent topics using topic modeling
- Comparing different approaches and evaluating their effectiveness

---

## Dataset

We use the **News Category Dataset** from Kaggle:

- [News Category Dataset on Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset)

Alternative datasets (optional):
- BBC News Headlines (manually collected)
- Custom Turkish news headlines

---

## NLP Preprocessing Steps

The following steps were applied **in order**, each justified in the notebook:

1. **Lowercasing** – Converts all text to lowercase for normalization
2. **Tokenization** – Splits text into words or tokens
3. **Stopword Removal** – Eliminates common, less informative words (e.g., “and”, “the”)
4. **Lemmatization** – Reduces words to their root forms
5. *(Optional)* **POS Tagging** – Tags each token with its part of speech for deeper analysis

---

## Vectorization Techniques

We convert text to numerical form using two methods:

### CountVectorizer (Bag of Words)
- Simple, frequency-based
- Pros: Fast and interpretable
- Cons: Sparse and context-insensitive

### TF-IDF (Term Frequency-Inverse Document Frequency)
- Weighs down common terms, highlights important ones
- Pros: Captures term importance across the corpus
- Cons: Still doesn't capture semantics

Example outputs and performance comparisons are provided in the notebook.

---

## Sentiment Analysis

Sentiment classification is applied using:

- **TextBlob** and **VADER** for English headlines
- **Zemberek**, **BERTurk**, or other transformer-based models for Turkish

Outputs include:
- Positive / Neutral / Negative classification
- Brief summary and visual comparison of model performances

---

## Topic Modeling

We apply the following unsupervised topic modeling techniques:

- **LDA (Latent Dirichlet Allocation)**
- **NMF (Non-Negative Matrix Factorization)**

For each model:
- Extracted topics are assigned **meaningful labels**
- Top keywords per topic are analyzed and interpreted
- Visualizations of topic distributions are included

---

