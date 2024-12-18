The repository is split into 3 folders.

1) The dataset folder contains the ISOT data.

   >> The files 'train.csv', 'val.csv', and 'test.csv' contain the raw text data and labels.
   >> The files 'TRAIN_ONLY_FEATURES.csv', 'VAL_ONLY_FEATURES.csv', and 'TEST_ONLY_FEATURES.csv' contain the extracted features.
   >> The file 'Sentiment_Data.csv' contains an annotated corpus of words scored by intensity and valence.

2) The scripts folder contains all the code used to run our experiments.

   >> LLM_API_calls.py: prompts the LLM + saves the obtained responses
   >> LLM_analysis.py: computes analysis of the LLM performance
   >> process_data.py: loads the data and preprocesses it for Naive Bayes
   >> NB.py: simple Naive Bayes script used as a benchmark
   >> feature_extraction: process used to extract the simple features described in the report
   >> standardize_features: proportionally scales extracted features
   >> NER.py: process used to tabulate the number of detected entities
   >> feature_classifier: simple logistic regression classifier using extracted features

3) The LLM_data contains the results of the API calls. Further details contained within.
