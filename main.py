import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest

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
plt.savefig('images/histogram.png')
plt.close()

# Диаграмма "ящик с усами"
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_processed, x='Year', color='orange')
plt.title('Диаграмма "ящик с усами" для Year')
plt.xlabel('Year')
plt.savefig('images/boxplot.png')
plt.close()

# Круговая диаграмма
plt.figure(figsize=(8, 8))
category_counts = df_processed['Country'].value_counts()
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', colors=['cyan', 'magenta'])
plt.title('Круговая диаграмма для Country')
plt.savefig('images/pie_chart.png')
plt.close()

# One-hot кодирование всех категориальных признаков
categorical_columns = ['Country', 'Disease Name', 'Disease Category',
                       'Age Group', 'Gender', 'Treatment Type',
                       'Availability of Vaccines/Treatment']
df_encoded = pd.get_dummies(df_processed, columns=categorical_columns, drop_first=True)

# Тепловая карта корреляции
plt.figure(figsize=(50, 50))
corr_matrix = df_encoded.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Тепловая карта корреляции')
plt.savefig('images/heatmap.png')
plt.close()

# Диаграмма countplot с группировкой по двум номинативным признакам
plt.figure(figsize=(18, 16))
sns.countplot(data=df_processed, x='Disease Category', hue='Gender', palette='viridis')
plt.title('Диаграмма countplot для Disease Category и Gender')
plt.xlabel('Disease Category')
plt.ylabel('Количество')
plt.savefig('images/countplot.png')
plt.close()

# Срез данных (первые 200 значений)
subset = df_encoded['Prevalence Rate (%)'].dropna().head(200)

# Проверка на нормальность
stat, p_value = normaltest(subset)

# Вывод результатов
print(f"Statistics: {stat:.2f}, p-value: {p_value:.5f}")

if p_value > 0.05:
    print("Распределение нормальное (при уровне значимости 0.05)")
else:
    print("Распределение не является нормальным (при уровне значимости 0.05)")

# Визуализация данных (гистограмма)
plt.hist(subset, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.title(f'Histogram of Prevalence Rate (%)')
plt.xlabel('Prevalence Rate (%)')
plt.ylabel("Frequency")
plt.savefig('images/Histogram of Prevalence Rate (%)')

# Заполнение пропусков
for column in df_encoded.columns:
    if df_encoded[column].isnull().any():
        if df_encoded[column].dtype == 'int64':
            median_value = df_encoded[column].median()
            df_encoded[column].fillna(median_value, inplace=True)
            print(f"Пропуски в столбце '{column}' заполнены медианой: {median_value}")
        elif df_encoded[column].dtype == 'float64':
            mean_value = df_encoded[column].mean()
            df_encoded[column].fillna(mean_value, inplace=True)
            print(f"Пропуски в столбце '{column}' заполнены средним: {mean_value}")
        else:
            mode_value = df_encoded[column].mode()[0]
            df_encoded[column].fillna(mode_value, inplace=True)
            print(f"Пропуски в столбце '{column}' заполнены модой: {mode_value}")


print("\nПроверка пропусков после заполнения:")
print(df_encoded.isnull().sum().sum())

# Сохранение предобработанного
df_encoded.head(1000).to_csv('processed_data.csv', index=False)
