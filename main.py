import pandas as pd

df = pd.read_csv('GlobalHealthStatistics.csv')

print('First and last 5 lines')
print(df.head(5))
print(df.tail(5))

print('\nData types')
print(df.info())

print('\nData static info:')
print(df.describe())

df_no_dup = df.drop_duplicates()
print('\nКоличество строк до удаления дубликатов:', len(df))
print('Количество строк после удаления дубликатов:', len(df_no_dup))

df_processed = df_no_dup.rename(columns={'Average Treatment Cost (USD)': 'Average Treatment Cost ($$$)'})