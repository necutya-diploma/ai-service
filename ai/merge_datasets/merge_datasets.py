import pandas as pd

df1 = pd.read_csv('datasets/spam.csv', encoding='latin-1')
df2 = pd.read_csv('datasets/train.csv', encoding='latin-1')

data1 = df1.copy()  ## make a copy of the data
data1.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)

data1 = data1.rename(columns={"v1": "label", "v2": "text"})

data2 = df2.copy()
data2.drop(columns=["Id", "following", "followers", "actions", "is_retweet", "location", ], inplace=True)
data2 = data2.rename(columns={"Type": "label", "Tweet": "text"})
data2['label'] = data2['label'].map({'Spam': 'spam', 'Quality': 'ham'})

# Stack the DataFrames on top of each other
new_df = pd.concat([data1, data2], axis=0)

new_df.to_csv('datasets/spam_ham.csv', index=False)
