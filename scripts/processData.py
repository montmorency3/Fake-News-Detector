import pandas as pd
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

def loadData(use_lemmatization=False):
    fakeData = pd.read_csv("Fake.csv")
    trueData = pd.read_csv("True.csv")

    # Apply preprocessing (with lemmatization or stemming based on the flag)
    fakeData["body"] = fakeData["text"].apply(lambda x: preprocessText(x, use_lemmatization))  # The body is the text
    trueData["body"] = trueData["text"].apply(lambda x: preprocessText(x, use_lemmatization))  # The body is the text
    fakeData["title"] = fakeData["title"].apply(lambda x: preprocessText(x, use_lemmatization))  # Title remains as is
    trueData["title"] = trueData["title"].apply(lambda x: preprocessText(x, use_lemmatization))  # Title remains as is

    # Assign labels (0 for fake, 1 for true)
    fakeData["label"] = 0
    trueData["label"] = 1

    # Combine both datasets
    data = pd.concat([fakeData[['title', 'body', 'label']], trueData[['title', 'body', 'label']]])
    return data

def preprocessText(text, use_lemmatization=False):
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

    # Tokenize the text
    words = text.split()

    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    if use_lemmatization:
        # Lemmatization (using WordNetLemmatizer)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    else:
        # Stemming (using PorterStemmer)
        words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]

    # Rejoin words back in
