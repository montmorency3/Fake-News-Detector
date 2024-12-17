# code used to merge all the extracted dfs
# combines Body, Title, NER information

### NOT RELEVANT TO FINAL REPORT ###
### ONLY USED TO CREATE CSV FILES ###

import pandas as pd

tag = "Test"
source = tag + "_Features.csv"
title = tag + "_Title_Features.csv"
ner_source = "NER/" + tag.lower() + "_ner_matrix.csv"

#df = pd.read_csv("fake.csv")
#df_fake_features = pd.read_csv("Old data/Fake_Features_EDIT2.csv")
#df_fake_titles = pd.read_csv("Old data/Fake_Title_Features_EDIT2.csv")

NER = pd.read_csv(ner_source)
body_f = pd.read_csv(source)
title_f = pd.read_csv(title)

# converts title colnames
new_cols = {}
for colname in title_f.columns:
    new_cols[colname] = "Title_" + colname
title_f = title_f.rename(columns=new_cols)

#rows_to_drop = [6457]
#body_f = df2.drop(rows_to_drop)

print(f"NER matrix has {len(NER)} entries")
print(f"Body features has {len(body_f)} entries")
print(f"Title features has {len(title_f)} entries")

df_full = pd.concat([body_f,title_f,NER],axis=1)
df_full.to_csv(tag + "_Features_Full.csv",index=False)
