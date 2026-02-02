from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

texts = [
    "I love programming in Python",
    "Python is great for data science",
    "I hate bugs in code",
    "Debugging is very frustrating",
    "Machine learning is amazing",
    "Errors make coding difficult"
]

labels = [
    "Positive",
    "Positive",
    "Negative",
    "Negative",
    "Positive",
    "Negative"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

user_text = input("\nEnter text to classify: ")
user_vector = vectorizer.transform([user_text])
prediction = model.predict(user_vector)

print("Predicted Class:", prediction[0])
