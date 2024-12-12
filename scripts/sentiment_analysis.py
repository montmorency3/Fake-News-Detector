import numpy as np
import pandas as pd

def load_sentiment_data():

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
  title_valence = np.zeros(len(dataset))
  title_intensity = np.zeros(len(dataset))
  
  for i,entry in enumerate(dataset):

    # add code here to split data entry as title vs text
    title = entry[0]
    text = entry[1]

    # scores main text
    sent = text.split(" ")
    valence,intensity = score_sentence(sent)
    valence_scores[i] = valence
    intensity_scores[i] = intensity

    # scores title
    sent = title.split(" ")
    valence,intensity = score_sentence(sent)
    title_valence[i] = valence
    title_intensity[i] = intensity

  return valence_scores, intensity_scores, title_valence, title_intensity

    

    
