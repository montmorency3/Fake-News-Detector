import pandas as pd
import re
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from nltk import FreqDist
from tqdm import tqdm  # Import tqdm for progress bars

# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('averaged_perceptron_tagger_eng')

def preprocessText(text, use_lemmatization=False, from_true_article=False):
    text = text.lower()

    if from_true_article:
        text = re.sub(r'^.*? - ', '', text)

    pattern_1 = re.compile(r'// <!.*?// ]]>')
    pattern_2 = re.compile(r'// <!.*?// ]]&gt')
    text = pattern_1.sub('', text)
    text = pattern_2.sub('', text)

    text = re.sub(r'[,:;\\/\'\"“”’‘<^>%&#.?!(){}[\]]','',text)

    words = text.split()

    filter_words = ["image", "images", "Image", "Images", "via", "Via", "featured", "Featured", "Getty", "Photo", "photo", "by", "(VIDEO)", "[VIDEO]", "WATCH"]
    words = [word for word in words if word not in filter_words]
    words = [word for word in words if "@" not in word]
    words = [word for word in words if "https" not in word]

    if use_lemmatization:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    else:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]

    return " ".join(words)

def extract_entities(text):
    """
    Extract named entities from the text using NLTK's ne_chunk.
    Returns a dictionary of entity types and their counts.
    """
    # Tokenize and POS tag the text
    words = word_tokenize(text)
    tagged = pos_tag(words)
    
    # Perform named entity recognition
    chunked = ne_chunk(tagged)
    
    entity_counts = {}
    
    # Extract named entities
    for subtree in chunked:
        if isinstance(subtree, nltk.Tree):  # It is a named entity
            entity_type = subtree.label()
            if entity_type not in entity_counts:
                entity_counts[entity_type] = 0
            entity_counts[entity_type] += 1

    # Flatten the dictionary into a string of entity types for feature extraction
    return ' '.join([f"{key}_{value}" for key, value in entity_counts.items()])


def save_ner_matrix(data, filename):
    """
    Saves the NER feature matrix to a CSV file.
    """
    vectorizer = CountVectorizer()
    ner_matrix = vectorizer.fit_transform(data['ner_features'])
    ner_df = pd.DataFrame(ner_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Save the NER matrix to a CSV file
    ner_df.to_csv(filename, index=False)

def loadData(dataset, use_lemmatization=True):
    # Use your cleaned dataset files (train.csv, val.csv, test.csv) from the preprocessing step
    train_data = pd.read_csv("train.csv")
    val_data = pd.read_csv("val.csv")
    test_data = pd.read_csv("test.csv")

    train_data['content'] = train_data['title'] + ' ' + train_data['body']
    val_data['content'] = val_data['title'] + ' ' + val_data['body']
    test_data['content'] = test_data['title'] + ' ' + test_data['body']
    
    # Extract NER features with a progress bar
    tqdm.pandas(desc="Extracting NER features")
    train_data['ner_features'] = train_data['content'].progress_apply(extract_entities)
    val_data['ner_features'] = val_data['content'].progress_apply(extract_entities)
    test_data['ner_features'] = test_data['content'].progress_apply(extract_entities)

    save_ner_matrix(train_data, "train_ner_matrix.csv")
    save_ner_matrix(val_data, "val_ner_matrix.csv")
    save_ner_matrix(test_data, "test_ner_matrix.csv")

    X_train, y_train = train_data['content'], train_data["label"]
    X_val, y_val = val_data['content'], val_data['label']
    X_test, y_test = test_data['content'], test_data['label']

    return X_train, y_train, X_val, y_val, X_test, y_test, train_data, val_data, test_data

def main():
    # Load data with progress
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test, train_data, val_data, test_data = loadData(dataset='ISOT', use_lemmatization=True)

    # Feature extraction using CountVectorizer for text data
    print("Vectorizing text data...")
    vectorizer = CountVectorizer(max_features=10000, stop_words='english')
    X_train_text_vec = vectorizer.fit_transform(X_train)
    X_val_text_vec = vectorizer.transform(X_val)
    X_test_text_vec = vectorizer.transform(X_test)

    # Feature extraction for NER features (just using raw counts here)
    print("Vectorizing NER features...")
    ner_vectorizer = CountVectorizer(max_features=100, binary=True)
    X_train_ner_vec = ner_vectorizer.fit_transform(train_data['ner_features'])
    X_val_ner_vec = ner_vectorizer.transform(val_data['ner_features'])
    X_test_ner_vec = ner_vectorizer.transform(test_data['ner_features'])

    # Combine text and NER features into a single feature matrix
    from scipy.sparse import hstack
    X_train_combined = hstack([X_train_text_vec, X_train_ner_vec])
    X_val_combined = hstack([X_val_text_vec, X_val_ner_vec])
    X_test_combined = hstack([X_test_text_vec, X_test_ner_vec])

    # Initialize and train Naive Bayes classifier
    print("Training Naive Bayes classifier...")
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_combined, y_train)

    # Predictions
    print("Making predictions...")
    y_pred = nb_classifier.predict(X_test_combined)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    with open('output_metrics.txt', 'w') as f:
    # Write the accuracy
        f.write(f'Accuracy: {accuracy:.4f}\n')

        # Write the classification report
        report = classification_report(y_test, y_pred)
        f.write(report)

        # Optional: Also print to the console for real-time feedback
        print(f'Accuracy: {accuracy:.4f}')
        print(report)

if __name__ == "__main__":
    main()

