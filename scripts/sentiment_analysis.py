import numpy as np
import pandas as pd
from processData import loadData

def load_sentiment_corpus():
  return pd.read_csv("Sentiment_Data.csv")

def score_sentence(df,sent):
  valence = 0
  intensity = 0
  for w in sent:
    # looks for entry in database
    target = df[df['Word']==w]
    if len(target) == 0:
      continue
    # else, computes sentiment scores for new word
    valence += (target['Valence']).mean()
    intensity += (target['Intensity']).mean()
  return valence, intensity
    
def score_sentiment():

  # loads sentiment corpus
  corpus = load_sentiment_corpus() 

  # loads dataset of interest
  df = loadData(use_lemmatization=True)
  
  # assume text is encoded as a list of words
  valence_scores = np.zeros(len(df))
  intensity_scores = np.zeros(len(df))
  title_valence = np.zeros(len(df))
  title_intensity = np.zeros(len(df))

  for i in len(df):

    # add code here to split data entry as title vs text
    title = df['title'][i]
    text = df['body'][i]
    
    # scores main text
    sent = text.split(" ")
    valence,intensity = score_sentence(corpus,sent)
    valence_scores[i] = valence
    intensity_scores[i] = intensity

    # scores title
    sent = title.split(" ")
    valence,intensity = score_sentence(corpus,sent)
    title_valence[i] = valence
    title_intensity[i] = intensity

  return valence_scores, intensity_scores, title_valence, title_intensity
