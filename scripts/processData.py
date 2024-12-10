import pandas as pd


def loadData():
    data = pd.read_csv("../dataset/Fake.csv")
    print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    return data

if __name__ == "__main__":
    loadData()