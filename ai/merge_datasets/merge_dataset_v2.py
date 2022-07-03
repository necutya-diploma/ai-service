import unicodedata

import pandas as pd
from bs4 import BeautifulSoup

df1 = pd.read_csv('datasets/spam_ham.csv', encoding='utf-8')

df2 = pd.read_csv('datasets/Youtube04-Eminem.csv', encoding='utf-8')
df2 = pd.concat([df2, pd.read_csv('datasets/Youtube01-Psy.csv', encoding='utf-8')], axis=0)
df2 = pd.concat([df2, pd.read_csv('datasets/Youtube02-KatyPerry.csv', encoding='utf-8')], axis=0)
df2 = pd.concat([df2, pd.read_csv('datasets/Youtube03-LMFAO.csv', encoding='utf-8')], axis=0)
df2 = pd.concat([df2, pd.read_csv('datasets/Youtube05-Shakira.csv', encoding='utf-8')], axis=0)

data1 = df1.copy()  ## make a copy of the data
data1['text'] = data1['text'].map(lambda x: str(x))
data1['text'] = data1['text'].map(lambda x: BeautifulSoup(x, "lxml").text)
data1['text'] = data1['text'].map(lambda x: unicodedata.normalize("NFKD", x))
data1['text'] = data1['text'].map(lambda x: x.encode('ascii', 'ignore').decode('utf-8'))

data2 = df2.copy()
data2.drop(columns=["COMMENT_ID", "AUTHOR", "DATE"], inplace=True)
data2 = data2.rename(columns={"CLASS": "label", "CONTENT": "text"})
data2['text'] = data2['text'].map(lambda x: str(x))
data2['label'] = data2['label'].map({1: 'spam', 0: 'ham'})
data2['text'] = data2['text'].map(lambda x: BeautifulSoup(x, "lxml").text)
data2['text'] = data2['text'].map(lambda x: unicodedata.normalize("NFKD", x))
data2['text'] = data2['text'].map(lambda x: x.encode('ascii', 'ignore').decode('utf-8'))

# Stack the DataFrames on top of each other
new_df = pd.concat([data1, data2], axis=0)

new_df.to_csv('datasets/spam_ham_v2.csv', index=False)

