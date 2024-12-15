from processData import loadData
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score


def trainNB():
    train_data, val_data, test_data = loadData()

    X_train = train_data['content']
    y_train = train_data["label"]
    X_test = test_data['content']
    y_test = test_data['label']

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Adjust max_features as needed
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train a Naive Bayes classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_tfidf, y_train)

    # Make predictions on the test set
    y_pred = nb_classifier.predict(X_test_tfidf)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    scores = cross_val_score(nb_classifier, X_train_tfidf, y_train, cv=5)
    print(f"Cross-validation mean accuracy: {scores.mean()}")



if __name__ == "__main__":
    trainNB()
