import numpy as np
import pandas as pd
import nltk
from nltk.corpus import words

# counts the number of misspelled words in example
# only for non-entities
# takes as input a dataset (list of string entries)

def count_errors(sentence):
  total = 0
  for w in sentence:
    if w in words.words():
      total += 1
  return total

def count_words(sentence):
  return len(set(sentence)) / len(sentence)

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
    is c.isnumeric():
      numeric += 1
    elif c.isupper():
      capitals += 1
  return punctuation, tags, citations, numeric, capitals
  
def derive_features(dataset):
  punctuation = np.zeros(len(dataset))
  tags = np.zeros(len(dataset))
  citations = np.zeros(len(dataset))
  numeric = np.zeros(len(dataset))
  capitals = np.zeros(len(dataset))
  word_counts = np.zeros(len(dataset))
  errors = np.zeros(len(dataset))
  word_lengths = np.zeros((3,len(dataset)))
  
  for i,sent in enumerate(dataset):

    punctuation[i], tags[i], citations[i], numeric[i], capitals[i] = count_symbols(sent) # tracks symbol counts
    
    sent = sent.split(" ")
    capitals[i] / len(sent) # adjusts capital letters against article size
    errors[i] = count_errors(sent) # counts misspellings
    word_counts[i] = count_words(sent) # counts percent unique words
    word_lengths[:,i] = count_words_lengths(sent)

  data_dict = {'Punctuation':punctuation,
              'Tags':tags,
              'Citations':citations,
              'Numeric':numeric,
              'Capitals':capitals,
              'Unique_Word_Prop':word_counts,
              'Misspellings':errors,
              'Short_Words':word_lengths[0,:],
              'Medium_Words':word_lengths[1,:],
              'Long_Words':word_lengths[2,:]}
  df = pd.DataFrame(data_dict)
  df.to_csv('Text_Dataset.csv', index=False)
  
  return df


