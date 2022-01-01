import pandas as pd
from glob import glob

df = pd.DataFrame()
for csv_ in glob('data/*.csv'):
    df = df.append(pd.read_csv(csv_))

df.set_index('date', inplace=True)
df.to_csv('combined_data.csv')
