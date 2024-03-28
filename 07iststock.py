import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini yükle
df = pd.read_csv('iststockoriginal.csv')

# Veri setinin ilk birkaç satırını görüntüle
print(df.head())

# Veri setinin betimsel istatistiklerini görüntüle
print(df.describe())

# Veri setindeki eksik değerleri kontrol et
print(df.isnull().sum())



# Tarih-zaman sütununu uygun bir şekilde dönüştür
df['Date'] = pd.to_datetime(df['Date'])


# Standartlaştırma
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# 'Symbol' sütununu düşürmeden önce yalnızca sayısal sütunları içeren bir alt veri çerçevesi oluştur
numeric_df = df[numeric_columns]

# Korelasyon matrisini hesapla
correlation_matrix = numeric_df.corr()
# Yeni özellikler oluştur (opsiyonel)

# Histogramlar
df.hist(figsize=(12, 10))
plt.show()





# Korelasyon matrisini hesapla
#correlation_matrix = df.drop(['Symbol'], axis=1).corr()


sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Kutu grafiği
sns.boxplot(data=df)
plt.title('Box Plot')
plt.show()

# Keman grafiği
sns.violinplot(data=df)
plt.title('Violin Plot')
plt.show()

# Scatter plot (Giriş vs. Çıkış)
sns.scatterplot(x='Open', y='Predict', data=df)
plt.title('Open vs. Predict')
plt.show()

# Normalleştirilmiş mutlak çapraz-korelasyon haritası
sns.heatmap(df.drop(['Symbol'], axis=1).corr().abs(), annot=True, cmap='coolwarm')
plt.title('Normalleştirilmiş Mutlak Çapraz-Korelasyon Haritası')
plt.show()
