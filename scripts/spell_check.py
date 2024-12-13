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

def spellcheck(dataset):
  errors = np.zeros(len(dataset))
  for i,sent in enumerate(dataset):
    errors[i] = count_errors(sent.split(" "))
  return errors
