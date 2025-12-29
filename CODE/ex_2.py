import nltk
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import math

# Download tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')

# -------------------------------
# Sample corpus
# -------------------------------
documents = [
    "I love natural language processing",
    "Natural language processing is very interesting",
    "I love machine learning",
    "Machine learning and deep learning"
]

query = "natural language"

# -------------------------------
# 1. TF (Term Frequency) Ranking
# -------------------------------
def tf_ranking(docs, query):
    scores = []
    query_terms = query.split()
    for doc in docs:
        words = doc.split()
        tf = Counter(words)
        score = sum(tf[q] for q in query_terms)
        scores.append(score)
    return scores

print("\nTF Ranking:", tf_ranking(documents, query))

# -------------------------------
# 2. TF-IDF Ranking
# -------------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents + [query])

query_vector = tfidf_matrix[-1]
doc_vectors = tfidf_matrix[:-1]

tfidf_scores = cosine_similarity(query_vector, doc_vectors)[0]
print("TF-IDF Ranking:", tfidf_scores)

# -------------------------------
# 3. Cosine Similarity (Bag of Words)
# -------------------------------
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents + [query])

query_vec = count_matrix[-1]
doc_vecs = count_matrix[:-1]

cos_scores = cosine_similarity(query_vec, doc_vecs)[0]
print("Cosine Similarity Ranking:", cos_scores)

# -------------------------------
# 4. PMI Ranking
# -------------------------------
def pmi(word, doc, corpus):
    total_docs = len(corpus)
    word_count = sum(1 for d in corpus if word in d)
    doc_count = 1 if word in doc else 0
    if doc_count == 0:
        return 0
    return math.log2((doc_count * total_docs) / word_count)

pmi_scores = []
for doc in documents:
    score = sum(pmi(w, doc, documents) for w in query.split())
    pmi_scores.append(score)

print("PMI Ranking:", pmi_scores)

# -------------------------------
# 5. Word2Vec Ranking
# -------------------------------
tokenized_docs = [nltk.word_tokenize(doc.lower()) for doc in documents]
model = Word2Vec(tokenized_docs, vector_size=50, window=3, min_count=1, workers=2)

def sentence_vector(sentence):
    words = nltk.word_tokenize(sentence.lower())
    vectors = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vectors, axis=0)

query_vec = sentence_vector(query)

w2v_scores = []
for doc in documents:
    doc_vec = sentence_vector(doc)
    score = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
    w2v_scores.append(score)

print("Word2Vec Ranking:", w2v_scores)

# -------------------------------
# Final Ranking Output
# -------------------------------
print("\nDocuments Ranked by TF-IDF:")
ranked = sorted(zip(documents, tfidf_scores), key=lambda x: x[1], reverse=True)
for doc, score in ranked:
    print(f"{score:.4f} -> {doc}")
