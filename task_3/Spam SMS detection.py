import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('spam.csv', encoding='latin1')
data['v2'] = data['v2'].str.lower()

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(data['v2'])
y = data['v1'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

user_input = input("Enter a message: ")
user_input = user_input.lower()
user_input_vectorized = tfidf_vectorizer.transform([user_input])

prediction = svm_classifier.predict(user_input_vectorized)

if prediction[0] == 0:
    print("The message is not a spam.")
else:
    print("The message is a spam.")
