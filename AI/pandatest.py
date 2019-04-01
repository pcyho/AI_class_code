import pandas as pd

df = pd.read_csv('watermelon_4_3.csv')
data = df.values[:, 1:].tolist()
labels = df.columns.values[1:-1].tolist()
print(data)
print(labels)
