import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# final classifier - skeleton code
# fix validation for hyperparams? regularize?

isot = True
base = "_FULL.csv"
liar_f = "LIAR2_FULL.csv"

include_title = True
include_ner = True
include_body = True
include_sentiment = True
include_char = True

sentiment_labs = ['Avg_Pos_Score','Avg_Neg_Score','Avg_Intensity']
ner_labs = ['gpe','organization','person']
title_labs = ['Title_Punctuation','Title_Tags',
                    'Title_Citations','Title_Numeric',
                    'Title_Unique_Words',
                    'Title_Short_Words','Title_Medium_Words',
                    'Title_Long_Words','Title_Positive_Score',
                    'Title_Negative_Score','Title_Intensity_Score',
                    'Title_Average_Word_Length']
char_labs = ['Punctuation','Tags','Citations','Avg_Capitals','Avg_Numeric']

to_drop = ['label','Text','Title_Text','Title_Capitals']


def load_data(filename,title=True,NER=True):
    df = pd.read_csv(filename)
    X = df.drop(to_drop,axis=1)
    if not title:
        X = X.drop(title_labs,axis=1)
    if not NER:
        X = X.drop(ner_labs,axis=1)
    if not include_sentiment:
        X = X.drop(sentiment_labs,axis=1)
    if not include_char:
        X = X.drop(char_labs,axis=1)
    features = X.columns
    #print(X.columns)
    X = scale(np.array(X))
    y = np.array(df['label'])
    return X,y,features

def load_liar_dataset():
    df = pd.read_csv(liar_f)
    X = df.drop(['label','Text'],axis=1)
    cols = X.columns
    X = scale(np.array(X))
    y = np.array(df['label'])
    # splits into train / val / test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
    X_val, X_test, y_val, y_test = train_test_split(X_test,y_test,test_size=0.40)
    return X_train, X_val, X_test, y_train, y_val, y_test, cols

# ISOT dataset loaders
if isot:
    X_train, y_train, features = load_data("TRAIN" + base,
                                           title=include_title,NER=include_ner)
    X_val, y_val, f = load_data("VAL" + base,
                                title=include_title,NER=include_ner)
    X_test, y_test, f = load_data("TEST" + base,
                                  title=include_title,NER=include_ner)

# LIAR2 dataset loaders
else:
    X_train, X_val, X_test, y_train, y_val, y_test, features = load_liar_dataset()

print(f"{X_train.shape=}, {X_val.shape=}, {X_test.shape=}")

feature_list = list(features)

# train a simple classifier
log_clf = LogisticRegression()
#clf.fit(X_val,y_val)
log_clf.fit(X_train,y_train)
acc_log = log_clf.score(X_test,y_test)
print(f"Simple logistic regression has acc {acc_log}")

info_dict = {'Feature':feature_list,
             'Coefficient':log_clf.coef_[0]}
info_df = pd.DataFrame(info_dict)

# svm classifier
# reg_factors = np.array([1,0.5,0.1,0.05,0.01,0.005,0.001])
# degrees = np.arange(3,8,1)
# ACCURACIES = np.zeros((len(degrees),len(reg_factors)))
# for i,reg in tqdm(enumerate(reg_factors)):
#     for j,deg in enumerate(degrees):
#         clf= SVC(C=reg,degree=deg)
#         clf.fit(X_train,y_train)
#         ACCURACIES[j,i] = clf.score(X_val,y_val)
# max_ind = np.argmax(ACCURACIES)
# reg_ind, deg_ind = np.unravel_index(max_ind,ACCURACIES.shape)
# max_reg = reg_factors[reg_ind]
# max_deg = degrees[deg_ind]
# print(f"Using best regularization strength {max_reg} and degree {max_deg}")
# best_clf = SVC(C=max_reg,degree=max_deg)
# best_clf.fit(X_train,y_train)
# acc = best_clf.score(X_test,y_test)
# print(f"Best SVM model has test acc of {acc}")

# NB classifier [on text data]
# smooths = np.array([0,1,2,5,10,25,50,100,500,1000])
# ACCS = np.zeros(len(smooths))
# for i,smooth in tqdm(enumerate(smooths)):
#     NB= CategoricalNB(alpha=smooth)
#     NB.fit(X_train,y_train)
#     ACCS[i] = NB.score(X_val,y_val)
# max_ind = np.argmax(ACCS)
# max_smooth = smooths[max_ind]
# print(f"Using best smoothin factor {max_smooth}")
# best_NB = CategoricalNB(alpha=max_smooth)
# best_NB.fit(X_train,y_train)
# acc = best_NB.score(X_test,y_test)
# print(f"Best NB model has test acc of {acc}")
