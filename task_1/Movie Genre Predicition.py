import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('train_data.txt', sep=' ::: ', header=None, names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'], engine='python')
test_data = pd.read_csv('test_data.txt', sep=' ::: ', header=None, names=['ID', 'TITLE', 'DESCRIPTION'], engine='python')
train_data['TEXT'] = train_data['TITLE'] + ' ' + train_data['DESCRIPTION']
test_data['TEXT'] = test_data['TITLE'] + ' ' + test_data['DESCRIPTION']

X_train, X_test, y_train, y_test = train_test_split(train_data['TEXT'], train_data['GENRE'], test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

logistic_classifier = LogisticRegression()
logistic_classifier.fit(X_train_tfidf, y_train)
y_pred = logistic_classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

test_data_tfidf = tfidf_vectorizer.transform(test_data['TEXT'])
test_predictions = logistic_classifier.predict(test_data_tfidf)

num_lines_to_print = 50  # You can adjust this number
for i in range(num_lines_to_print):
    print(f'Test Data ID: {test_data["ID"].iloc[i]}, Predicted Genre: {test_predictions[i]}')
