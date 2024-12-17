import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# final classifier - skeleton code
# fix validation for hyperparams? regularize?

def load_data(filename):
    df = pd.read_csv(filename)
    X = df.drop('Label',axis=1)
    y = df['Label']
    return X,y

X_train, y_train = load_data("Merged_Data_Train.csv")
X_val, y_val = load_data("Merged_Data_Val.csv")
X_test, y_test = load_data("Merged_Data_Test.csv")

# train a simple classifier
clf = LogisticRegression()
clf.fit(X_train,y_train)
acc = clf.score(X_test,y_test)