# text_preprocessing.py

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

from torchnlp.word_to_vector import GloVe

# ---------------------------
# Sample Input Text
# ---------------------------
text = """
Natural Language Processing (NLP) is a subfield of Artificial Intelligence.
It focuses on the interaction between computers and humans using natural language.
"""

# ---------------------------
# 1. Word Tokenization (NLTK)
# ---------------------------
tokens = word_tokenize(text.lower())
print("Tokens:")
print(tokens)

# ---------------------------
# 2. Punctuation Removal
# ---------------------------
tokens_no_punct = [word for word in tokens if word not in string.punctuation]
print("\nAfter Punctuation Removal:")
print(tokens_no_punct)

# ---------------------------
# 3. Stop Word Removal
# ---------------------------
stop_words = set(stopwords.words("english"))
filtered_words = [word for word in tokens_no_punct if word not in stop_words]
print("\nAfter Stop Word Removal:")
print(filtered_words)

# ---------------------------
# 4. Topical Word Extraction
#    (Most Frequent Words)
# ---------------------------
word_freq = Counter(filtered_words)
topical_words = word_freq.most_common(5)

print("\nTopical Words (Most Frequent):")
print(topical_words)


