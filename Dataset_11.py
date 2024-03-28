import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Veri kümesini yükle
df = pd.read_csv('accident_news.csv')

# Veri ön işleme
# Veri kümesinin başını kontrol et
print(df.head())

# Veri kümesinin bilgilerini kontrol et
print(df.info())

# Sayısal olmayan sütunları belirle
non_numeric_columns = df.select_dtypes(exclude=['number']).columns
print("Sayısal olmayan sütunlar:", non_numeric_columns)

# Sayısal olmayan sütunları veri kümesinden kaldır
df = df.drop(columns=non_numeric_columns)

# Standartlaştırma
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Yeni özellikler çıkarımı
# Burada gerekli feature engineering işlemlerini gerçekleştirebilirsiniz

# Histogramlar
for column in df.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], bins=20, kde=True, color='skyblue')
    plt.title(f'{column} Dağılımı')
    plt.xlabel(column)
    plt.ylabel('Frekans')
    plt.grid(True)
    plt.show()

# Kutu Grafiği (Box Plot)
for column in df.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[column], color='skyblue')
    plt.title(f'{column} Kutu Grafiği')
    plt.ylabel(column)
    plt.grid(True)
    plt.show()

# Violin Grafiği
for column in df.columns:
    plt.figure(figsize=(8, 6))
    sns.violinplot(y=df[column], color='skyblue')
    plt.title(f'{column} Violin Grafiği')
    plt.ylabel(column)
    plt.grid(True)
    plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Death', y='Injured', color='skyblue')
plt.title('Death vs. Injured Scatter Plot')
plt.xlabel('Death')
plt.ylabel('Injured')
plt.grid(True)
plt.show()

# Korelasyon Matrisi
correlation_matrix = df_standardized.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(np.abs(correlation_matrix), annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title('Normalize Edilmiş Mutlak Çapraz Korelasyon Haritası')
plt.show()
