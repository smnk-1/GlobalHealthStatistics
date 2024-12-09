import pandas as pd

df = pd.read_csv('GlobalHealthStatistics.csv')

print('First and last 5 lines')
print(df.head(5))
print(df.tail(5))