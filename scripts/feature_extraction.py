import numpy as np
import pandas as pd
import nltk
from nltk.corpus import words
from collections import Counter
import re
from nltk.stem import WordNetLemmatizer as wnl
from tqdm import tqdm

# counts the number of misspelled words in example
# only for non-entities
# takes as input a dataset (list of string entries)

def count_errors(sentence):
    total = 0
    for w in sentence:
        if w not in words.words():
            total += 1
    return total

def count_words(sentence):
    return len(set(sentence)), len(sentence)

def avg_word_length(sentence):
    avg_len = 0
    if len(sentence) != 0:
        for w in words:
            avg_len += len(w)
        avg_len = avg_len / len(sentence)
    return avg_len

def count_words_lengths(sentence):
    counts = np.zeros(3) # small, medium, large
    for w in sentence:
        if len(w) <= 6:
            counts[0] += 1
        elif len(w) <=10:
            counts[1] += 1
        else:
            counts[2] += 1
    return counts

def count_symbols(text):
  
    count = Counter(text)
    # counts all punctuation symbols
    punctuation = count["!"] # highly expressive
    punctuation += count["?"]
    tags = count["#"] # hashtags or mentions
    tags += count["@"] 
    citations = count["\""] / 2 # quotes
    # counts numeric symbols
    numeric = 0
    capitals = 0
    for c in text:
        if c.isnumeric():
            numeric += 1
        elif c.isupper():
            capitals += 1
    return punctuation, tags, citations, numeric, capitals

def load_sentiment_corpus():
  return pd.read_csv("Sentiment_Data.csv")

def score_sentiment(df,sent):
    pos = 0
    neg = 0
    intensity = 0
    for w in sent:
        # looks for entry in database
        target = df[df['Word']==w]
        if len(target) == 0:
            continue
        # else, computes sentiment scores for new word
        valence = (target['Valence']).mean()
        if valence >= 0:
            pos += valence
        else:
            neg -= valence
        intensity += (target['Intensity']).mean()
    return pos, neg, intensity
    
# def score_sentiment():
#   
#     # assume text is encoded as a list of words
#   
#     title_pos = np.zeros(len(df))
#     title_neg = np.zeros(len(df))
#     title_intensity = np.zeros(len(df))
# 
#     for i in len(df):
#         # add code here to split data entry as title vs text
#         title = df['title'][i]
#         text = df['body'][i]
#     
#         # scores main text
#     sent = text.split(" ")
#     pos,neg,intensity = score_sentence(corpus,sent)
#     pos_scores[i] = pos
#     neg_scores[i] = neg
#     intensity_scores[i] = intensity
# 
#     # scores title
#     sent = title.split(" ")
#     pos,neg,intensity = score_sentence(corpus,sent)
#     title_pos[i] = pos
#     title_neg[i] = neg
#     title_intensity[i] = intensity
# 
#   return valence_scores, intensity_scores, title_valence, title_intensity


def derive_symbol_features(dataset,data_label):
    # before preprocessing >> extract direct char features
    punctuation = np.zeros(len(dataset))
    tags = np.zeros(len(dataset))
    citations = np.zeros(len(dataset))
    numeric = np.zeros(len(dataset))
    capitals = np.zeros(len(dataset))
    # inspects at character level
    for i,sent in tqdm(enumerate(dataset)):
        punctuation[i], tags[i], citations[i], numeric[i], capitals[i] = count_symbols(sent) # tracks symbol counts
    # generates CSV file
    data_dict = {'Punctuation':punctuation,
              'Tags':tags,
              'Citations':citations,
              'Numeric':numeric,
              'Capitals':capitals}
    df = pd.DataFrame(data_dict)
    #df.to_csv(data_label + '_Symbol_Features.csv', index=False)
    return df

def process_sentence(text,lemmatizer):
    # removes bad char
    sent = re.sub(r'[,:*;\\/\'\"<>@%&#0123456789.?!(){}[\]]','',text)
    sent = sent.lower()
    tokens = sent.split()
    processed = []
    for w in tokens:
        processed.append(lemmatizer.lemmatize(w))
    return processed
  
def derive_word_features(dataset,data_label):
    # requires lemmatized inputs
    # remove stopwords first? no...
    # remove numeric char
    # remove strange punctuation
    processed_strings = []
    unique_words = np.zeros(len(dataset))
    sent_length = np.zeros(len(dataset))
    errors = np.zeros(len(dataset))
    word_lengths = np.zeros((3,len(dataset)))
    avg_lengths = np.zeros(len(df))
    pos_scores = np.zeros(len(dataset))
    neg_scores = np.zeros(len(dataset))
    intensity = np.zeros(len(dataset))
    
    lemmatizer = wnl() # uses lemmatizers to process
    df_sentiment = load_sentiment_corpus() # to score sentiment
  
    for i,sentence in tqdm(enumerate(dataset)):
        sent = process_sentence(sentence,lemmatizer)
        processed_strings.append(sent)
        #capitals[i] / len(sent) # adjusts capital letters against article size
        #errors[i] = count_errors(sent) # counts misspellings
        unique_words[i], sent_length[i] = count_words(sent) # counts percent unique words
        word_lengths[:,i] = count_words_lengths(sent)
        avg_lengths[i] = avg_word_length(sent)
        pos_scores[i], neg_scores[i], intensity[i] = score_sentiment(df_sentiment,sent)
        
    data_dict = {'Unique_Words':unique_words,
                 'Word_Counts':sent_length,
              #'Misspellings':errors,
              'Short_Words':word_lengths[0,:],
              'Medium_Words':word_lengths[1,:],
              'Long_Words':word_lengths[2,:],
                 'Average_Word_Lengths':avg_lengths,
                 'Positive_Score':pos_scores,
                 'Negative_Score':neg_scores,
                 'Intensity_Score':intensity}
    df = pd.DataFrame(data_dict)
    #df.to_csv(data_label + '_Word_Features.csv', index=False)
    text_dict = {'Text':processed_strings}
    return pd.DataFrame(text_dict), df

sample_corpus = ["This is the first sentence!",
                 "Morpologically More Complexified Sentence According To Philantropist",
                 "I hate you, gross ass!!!",
                 "Numeric text tienes 35921 or maybe just 6 digits #winning",
                 "I love to quote ppl like this \"love you honey buns\""]

### MAIN CODE ###

dataframe = pd.read_csv("True.csv")
data = list(dataframe['text'])
print(data[1:10])
data_label = 'True_v1'
df_sym = derive_symbol_features(data,data_label)
df_sent, df_word = derive_word_features(data,data_label)
df = pd.concat([df_sent,df_sym,df_word],axis=1)
df.to_csv(data_label + '_Features.csv', index=False)
