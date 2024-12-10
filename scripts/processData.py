import pandas as pd
from nltk.corpus import stopwords
import re
from nltk import PorterStemmer

def loadData():
    fakeData = pd.read_csv("Fake.csv")
    trueData = pd.read_csv("True.csv")

    fakeData["content"] = fakeData["title"] + fakeData["text"]
    trueData["content"] = trueData["title"] + trueData["text"]

    fakeData["content"] = fakeData["content"].apply(preprocessText)
    trueData["content"] = trueData["content"].apply(preprocessText)


    fakeData["label"] = 0
    trueData["label"] = 0

    data = pd.concat([fakeData[['content', 'label']], trueData[['content', 'label']]])
    return data


def preprocessText(text):
    stemmer = PorterStemmer()
    """
    Preprocess the text by performing the following steps:
    - Lowercasing
    - Removing special characters and digits
    - Removing stop words
    - Optional: Lemmatization or stemming
    """
    # Lowercase the text
    text = text.lower()

    # Remove non-alphanumeric characters (keep spaces)

    # Tokenize and stem each word
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
    print(text)
    # Rejoin words back into a single string
    return " ".join(words)

if __name__ == "__main__":

    data = loadData()
    print(data.head())