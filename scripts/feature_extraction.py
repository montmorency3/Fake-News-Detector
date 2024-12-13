import numpy as np
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
  
def derive_features(dataset):
  word_counts = np.zeros(len(dataset))
  errors = np.zeros(len(dataset))
  for i,sent in enumerate(dataset):

    sent = sent.split(" ")
    errors[i] = count_errors(sent) # counts misspellings
    word_counts[i] = count_words(sent) # counts percent unique words
  return errors
