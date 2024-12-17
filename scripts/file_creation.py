### NOT IMPORT FOR ANALYSIS ###
### SIMPLY TO ADJUST THE DATASET IN FILES ###

import pandas as pd

# 1 = True, 0 = Fake

out_f = "Train_Title_Features.csv"
in_f = "train.csv"

df = pd.read_csv(in_f)

df_true = pd.read_csv("True_Title_Features_EDIT2.csv")
df_fake = pd.read_csv("Fake_Title_Features_EDIT2.csv")
df_full = pd.concat([df_true,df_fake],axis=0)

print(f"{len(df_true)} true instances, {len(df_fake)} fakes")
print(f"In total, {len(df_full)} instances!")

bias = len(df_true)
order = []
labels = []

for i in range(len(df)):
    txt_id = df['id'][i]
    label = 1
    
    if not df['label'][i]: # fake instance : add bias
        txt_id += bias
        label = 0
    
    order.append(txt_id) # reorders df
    labels.append(label)
    
new_df = df_full.iloc[order]
# lab_dict = {'Label':labels}
# lab_df = pd.DataFrame(lab_dict)
# new_df = pd.concat([new_df,lab_df],axis=1)
new_df.to_csv(out_f, index=False)
