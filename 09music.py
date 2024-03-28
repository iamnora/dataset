import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükle
df = pd.read_csv('Music Informations and Lyrics_ from Spotify and Musixmatch.csv')

# 'Date' sütununun varlığını kontrol et
if 'Date' in df.columns:
    # 'Date' sütununu veri çerçevesinden çıkar
    df.drop(['Date'], axis=1, inplace=True)


# Eksik değerleri doldur
numeric_columns = df.select_dtypes(include=np.number).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

categorical_columns = df.select_dtypes(exclude=np.number).columns
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Veri setini CSV dosyasına dönüştür
df.to_csv("veriseti.csv", index=False)

# Standardizasyon
df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()

# Yeni özellikler çıkar
# Örneğin, "Release Date" sütunundan yıl bilgisini çıkarabiliriz
# Tarih verilerini doğru formata dönüştürmek için try-except bloğu kullanalım
df['Release Year'] = pd.to_datetime(df['Release Date'], errors='coerce').dt.year



# Histogramlar
df.hist(figsize=(15, 10))
plt.show()

# Korelasyon matrisi
correlation_matrix = df.drop(['Release Date', 'Track Name', 'Artist', 'Artist Genres', 'Release Date', 'Lyrics'], axis=1).corr()

# Korelasyon matrisinin görselleştirilmesi
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix.abs(), annot=True, cmap='coolwarm')
plt.show()

# Box plot
df.boxplot(figsize=(15, 10))
plt.show()

# Violin plot
plt.figure(figsize=(15, 10))
sns.violinplot(data=df, inner="point")
plt.show()

# Scatter plot
sns.pairplot(df, diag_kind='hist')
plt.show()
