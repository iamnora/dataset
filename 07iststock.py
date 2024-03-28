import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini yükle
df = pd.read_csv('isedataset.csv')

# Veri setinin ilk birkaç satırını görüntüle
print(df.head())

# Veri setinin betimsel istatistiklerini görüntüle
print(df.describe())

# Veri setindeki eksik değerleri kontrol et
print(df.isnull().sum())

# Standartlaştırma
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Predict']
df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()

# Yeni özellikler oluştur (opsiyonel)

# Histogramlar
df.hist(figsize=(12, 10))
plt.show()

# Korelasyon matrisi
# Korelasyon matrisini hesapla
# 'Date' sütununu veri setinden çıkar
df.drop('Date', axis=1, inplace=True)

# Korelasyon matrisini hesapla
# Korelasyon matrisini hesapla
correlation_matrix = df.drop(['Symbol'], axis=1).corr()


sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Korelasyon Matrisi')
plt.show()

# Kutu grafiği
sns.boxplot(data=df)
plt.title('Kutu Grafiği')
plt.show()

# Keman grafiği
sns.violinplot(data=df)
plt.title('Keman Grafiği')
plt.show()

# Scatter plot (Giriş vs. Çıkış)
sns.scatterplot(x='Open', y='Predict', data=df)
plt.title('Open vs. Predict')
plt.show()

# Normalleştirilmiş mutlak çapraz-korelasyon haritası
sns.heatmap(df.drop(['Symbol'], axis=1).corr().abs(), annot=True, cmap='coolwarm')
plt.title('Normalleştirilmiş Mutlak Çapraz-Korelasyon Haritası')
plt.show()
