import pandas as pd
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter

from sklearn.model_selection import train_test_split

'''
Contains functions for preprocessing of the data, runtime is about 20-30 minutes. Results can be found in 
train.csv, val.csv and test.csv
'''


def cleanData(use_lemmatization=False):
    fakeData = pd.read_csv("../dataset/Fake.csv")
    trueData = pd.read_csv("../dataset/True.csv")

    # add id column to identify articles
    trueData["id"] = trueData['id'] = range(0, len(trueData))
    fakeData["id"] = fakeData['id'] = range(0, len(fakeData))

    # Assign labels (0 for fake, 1 for true)
    fakeData["label"] = 0
    trueData["label"] = 1

    # Apply preprocessing (with lemmatization or stemming based on the flag)
    fakeData["body"] = fakeData["text"].apply(lambda x: preprocessText(x, use_lemmatization))
    trueData["body"] = trueData["text"].apply(lambda x: preprocessText(x, use_lemmatization, True))
    fakeData["title"] = fakeData["title"].apply(lambda x: preprocessText(x, use_lemmatization))
    trueData["title"] = trueData["title"].apply(lambda x: preprocessText(x, use_lemmatization))


    # Combine both datasets
    data = pd.concat([fakeData[['id', 'title', 'body', 'label']], trueData[['id', 'title', 'body', 'label']]])

    # look for emoty/missing articles and titles
    data = data[data['body'].notna() & (data['body'] != '')]
    data = data[data['title'].notna() & (data['title'] != '')]

    # look for duplicated articles
    data = data.drop_duplicates(subset=['body'])
    data = data.drop_duplicates(subset=['title'])

    # split the data into train, test and validation set
    temp_data, test_data = train_test_split(data, test_size=0.025, random_state=42)
    train_data, val_data = train_test_split(temp_data, test_size=0.10, random_state=42)
    test_data.to_csv('../dataset/test.csv', columns=['id', 'label', 'title', 'body'] ,index=False)
    train_data.to_csv('../dataset/train.csv', columns=['id', 'label', 'title', 'body'], index=False)
    val_data.to_csv('../dataset/val.csv', columns=['id', 'label', 'title', 'body'], index=False)


def preprocessText(text, use_lemmatization=False, from_true_article=False):
    """
    Preprocess the text by performing the following steps:
    - Lowercasing
    - Removing special characters
    - Removing code sections, links, @mentions
    - Removing stop words
    - Optional: Lemmatization or stemming
    """
    # Lowercase the text
    text = text.lower()

    # True articles always start with "CITY-NAME (Reuters) - "
    if from_true_article:
        text = re.sub(r'^.*? - ', '', text)

    # remove code sections in text
    pattern_1 = re.compile(r'// <!.*?// ]]>')
    pattern_2 = re.compile(r'// <!.*?// ]]&gt')
    text = pattern_1.sub('', text)
    text = pattern_2.sub('', text)

    # Remove non-alphanumeric characters
    text = re.sub(r'[,:;\\/\'\"<^>%&#.?!(){}[\]]','',text)

    # Tokenize the text
    words = text.split()

    # Filter certain words: featured image, image via Getty,..., removing @mentions, removing links
    filter_words = ["image", "images", "Image", "Images", "via", "Via", "featured", "Featured", "Getty", "Photo",
                    "photo", "by", "(VIDEO)", "[VIDEO]", "WATCH"]
    words = [word for word in words if word not in filter_words]
    words = [word for word in words if "@" not in word]
    words = [word for word in words if "https" not in word]

    if use_lemmatization:
        # Lemmatization (using WordNetLemmatizer)
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    else:
        # Stemming (using PorterStemmer)
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]

    # Rejoin words back in
    return " ".join(words)

def loadData(use_lemmatization=True):
    # read processed version of dataset
    train_data = pd.read_csv("../dataset/train.csv")
    val_data = pd.read_csv("../dataset/val.csv")
    test_data = pd.read_csv("../dataset/test.csv")

    # combine title and article text
    train_data['content'] = train_data['title'] + ' ' + train_data['body']
    val_data['content'] = val_data['title'] + ' ' + val_data['body']
    test_data['content'] = test_data['title'] + ' ' + test_data['body']

    return train_data, val_data, test_data

def count_punctuation(text):

    count = Counter(text)
    total = 0

    # counts all punctuation symbols
    total += count["!"]
    total += count["?"]
    total += count["#"]
    total += count["@"]
    
    return total


if __name__ == "__main__":
    cleanData(use_lemmatization=True)