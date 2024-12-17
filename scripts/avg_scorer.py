import pandas as pd

#tag = "Val"
#filename = tag + "_Features_Full.csv"
# out_f = tag.upper() + "_FULL.csv"
out_f = "LIAR2_FULL.csv"
filename = "Liar_Dataset_Features_EDIT2.csv"
df = pd.read_csv(filename)

# standardizes certain features via WC
df['Avg_Pos_Score'] = df['Positive_Score'] / df['Word_Counts']
df['Avg_Neg_Score'] = df['Negative_Score'] / df['Word_Counts']
df['Avg_Intensity'] = df['Intensity_Score'] / df['Word_Counts']
df['Avg_Unique'] = df['Unique_Words'] / df['Word_Counts']
df['Avg_Capitals'] = df['Capitals'] / df['Word_Counts']
df['Avg_Numeric'] = df['Numeric'] / df['Word_Counts']
df['Avg_Short'] = df['Short_Words'] / df['Word_Counts']
df['Avg_Medium'] = df['Medium_Words'] / df['Word_Counts']
df['Avg_Long'] = df['Long_Words'] / df['Word_Counts']

#also drop 'Title_Word_Counts'
df2 = df.drop(['Positive_Score','Negative_Score',
               'Intensity_Score','Unique_Words',
               'Capitals','Numeric','Short_Words',
               'Medium_Words','Long_Words','Word_Counts',
               ],axis=1)

describ = df2.describe()
means = describ.loc['mean']

df2.to_csv(out_f,index=False)