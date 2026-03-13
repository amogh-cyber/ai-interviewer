import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------- Load Data --------
with open("questions.json", "r") as f:
    data = json.load(f)

# -------- Prepare Dataset --------
X, y = [], []
for role, questions in data.items():
    X += questions
    y += [role] * len(questions)

# -------- Vectorization + Training --------
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# -------- Test Prediction --------
test_question = ["How does React handle component state"]
test_vec = vectorizer.transform(test_question)
prediction = model.predict(test_vec)

print("Predicted Role:", prediction[0])
