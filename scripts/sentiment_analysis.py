import numpy as np
import pandas as pd

def load_sentiment_data():

  return pd.read_csv("Sentiment_Data.csv")

def score_sentiment(dataset):

  # preprocess dataset
  text_data = preprocessText(dataset)
  
  # loads pre-evaluated sentiment data
  df = load_sentiment_data()

  # given a bank of sentiment data, scores a piece of text
  # attributes a valence score and an intensity score

  # assume text is encoded as a list of words
  valence_scores = np.zeros(len(dataset))
  intensity_scores = np.zeros(len(dataset))
  
  for i,entry in enumerate(dataset):
    
    sent = entry.split(" ")
    
    for w in entry:

      # looks for entry in database
      target = df[df['Word']==w]
      if len(target) == 0:
        continue

      # else, computes sentiment scores for new word
      valence_scores[i] += target['Valence']
      intensity_scores[i] += target['Intensity']

  return valence_scores, intensity_scores

    

    
