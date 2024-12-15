from processData import loadData
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

# Ensure required NLTK resources are downloaded (required only once)
#nltk.download('punkt')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('averaged_perceptron_tagger_eng')
#nltk.download('maxent_ne_chunker_tab')

# Function to extract named entities using NLTK
def extract_named_entities(texts):
    entity_features = []
    for text in texts:
        # Tokenize and tag parts of speech
        tokens = word_tokenize(text)
        tagged_tokens = pos_tag(tokens)
        
        # Perform Named Entity Recognition (NER)
        tree = ne_chunk(tagged_tokens)
        
        # Extract named entities from the tree
        entities = []
        for subtree in tree:
            if isinstance(subtree, Tree):  # Check if it's a named entity
                entity = " ".join([word for word, tag in subtree])
                entities.append(entity)
        
        # Store the entities as a space-separated string
        entity_features.append(" ".join(entities))
    
    return entity_features

def trainNB(dataset):
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = loadData(dataset)

    # Extract NER features using NLTK
    print("Extracting NER features using NLTK...")
    X_train_entities = extract_named_entities(X_train)
    X_test_entities = extract_named_entities(X_test)
    
    # Vectorize the original text using TF-IDF
    print("Vectorizing original text...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Vectorize the NER features
    print("Vectorizing NER features...")
    entity_vectorizer = TfidfVectorizer()
    X_train_entities_tfidf = entity_vectorizer.fit_transform(X_train_entities)
    X_test_entities_tfidf = entity_vectorizer.transform(X_test_entities)
    
    # Combine text features and NER features
    print("Combining text and NER features...")
    from scipy.sparse import hstack
    X_train_combined = hstack([X_train_tfidf, X_train_entities_tfidf])
    X_test_combined = hstack([X_test_tfidf, X_test_entities_tfidf])
    
    # Train the Naive Bayes classifier
    print("Training Naive Bayes classifier...")
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_combined, y_train)
    
    # Make predictions and evaluate
    print("Evaluating model...")
    y_pred = nb_classifier.predict(X_test_combined)
    
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Cross-validation
    scores = cross_val_score(nb_classifier, X_train_combined, y_train, cv=5)
    print(f"Cross-validation mean accuracy: {scores.mean()}")

if __name__ == "__main__":
    trainNB('ISOT')
