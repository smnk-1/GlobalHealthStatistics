import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# Гистограмма распределения числового признака
plt.figure(figsize=(8, 6))
plt.hist(df_processed['Doctors per 1000'], bins=10, color='blue', alpha=0.7)
plt.title('Гистограмма распределения Doctors per 1000')
plt.xlabel('Значения')
plt.ylabel('Частота')
plt.savefig('histogram.png')
plt.close()

# Диаграмма "ящик с усами"
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_processed, x='Year', color='orange')
plt.title('Диаграмма "ящик с усами" для Year')
plt.xlabel('Year')
plt.savefig('boxplot.png')
plt.close()

# Круговая диаграмма
plt.figure(figsize=(8, 8))
category_counts = df_processed['Country'].value_counts()
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', colors=['cyan', 'magenta'])
plt.title('Круговая диаграмма для Country')
plt.savefig('pie_chart.png')
plt.close()

# # Тепловая карта корреляции
# plt.figure(figsize=(10, 8))
# corr_matrix = df_processed.corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Тепловая карта корреляции')
# plt.savefig('heatmap.png')
# plt.close()

# Диаграмма countplot с группировкой по двум номинативным признакам
# plt.figure(figsize=(10, 8))
# sns.countplot(data=df_processed, x='categorical_feature1', hue='categorical_feature2', palette='viridis')
# plt.title('Диаграмма countplot для categorical_feature1 и categorical_feature2')
# plt.xlabel('categorical_feature1')
# plt.ylabel('Количество')
# plt.savefig('countplot.png')
# plt.close()

print("Все графики построены и сохранены как PNG-файлы.")
