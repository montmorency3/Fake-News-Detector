import pandas as pd
from nltk.corpus import stopwords
import re
from nltk import PorterStemmer

def loadData():
    fakeData = pd.read_csv("Fake.csv")
    trueData = pd.read_csv("True.csv")

    fakeData["body"] = fakeData["text"].apply(preprocessText)  # The body is the text
    trueData["body"] = trueData["text"].apply(preprocessText)  # The body is the text
    fakeData["title"] = fakeData["title"].apply(preprocessText)  # Title remains as is
    trueData["title"] = trueData["title"].apply(preprocessText)  # Title remains as is

    # fakeData["content"] = fakeData["content"].apply(preprocessText)
    # trueData["content"] = trueData["content"].apply(preprocessText)


    fakeData["label"] = 0
    trueData["label"] = 1

    data = pd.concat([fakeData[['title', 'body', 'label']], trueData[['title', 'body', 'label']]])
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
    text = re.sub(r'[,:;\\/\'\"<>@%&#.?!(){}[\]]','',text)

    # Tokenize and stem each word
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
    print(text)
    # Rejoin words back into a single string
    return " ".join(words)

if __name__ == "__main__":

    data = loadData()
    print(data.head())
