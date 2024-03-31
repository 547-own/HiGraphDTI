import pandas as pd
with open('data.txt', 'r') as f:
    data_list = f.read().strip().split('\n')

data_split_list = []
for no,data in enumerate(data_list):
    data_split_list.append(data.strip().split())

df = pd.DataFrame(data_split_list)
df.columns = ['smiles', 'sequence', 'interaction']

df.to_csv('../human.csv', index=None)