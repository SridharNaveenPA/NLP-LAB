import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# Documents and query
documents = [
    "I love natural language processing",
    "Natural language processing is very interesting",
    "I love machine learning",
    "Machine learning and deep learning"
]

query = "natural language"

# ---------------- TF RANKING ----------------
def tf_ranking(docs, query):
    query_terms = query.lower().split()
    scores = []

    for doc in docs:
        words = doc.lower().split()
        tf = Counter(words)
        score = sum(tf[term] for term in query_terms)
        scores.append(score)

    return scores

tf_scores = tf_ranking(documents, query)

print("\nTF Ranking:")
for doc, score in zip(documents, tf_scores):
    print(f"{score} -> {doc}")

# ---------------- TF-IDF RANKING ----------------
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents + [query])

query_vector = tfidf_matrix[-1]
doc_vectors = tfidf_matrix[:-1]

tfidf_scores = cosine_similarity(query_vector, doc_vectors)[0]

print("\nTF-IDF Ranking:")
for doc, score in zip(documents, tfidf_scores):
    print(f"{score:.4f} -> {doc}")

# ---------------- COSINE SIMILARITY (COUNT VECTORS) ----------------
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents + [query])

query_vec = count_matrix[-1]
doc_vecs = count_matrix[:-1]

cosine_scores = cosine_similarity(query_vec, doc_vecs)[0]

print("\nCosine Similarity Ranking:")
for doc, score in zip(documents, cosine_scores):
    print(f"{score:.4f} -> {doc}")
