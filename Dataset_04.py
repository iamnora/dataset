import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Veri kümesini yükle
dataset = pd.read_csv('imdb_reviews_ratings.csv', sep=',')

# Sayısal olmayan sütunları çıkar
non_numeric_columns = ['movie_id', 'movie_title', 'genres', 'review_id', 'review']
dataset_numeric = dataset.drop(columns=non_numeric_columns)

# Tür listesini sayısal değerlere dönüştürme (One-hot encoding)
genres = dataset['genres'].str.strip("[]").str.replace("'", "").str.split(", ", expand=True)
genre_columns = pd.get_dummies(genres.apply(pd.Series).stack()).sum().reset_index(drop=True)
dataset_numeric = pd.concat([dataset_numeric, genre_columns], axis=1)

# Review sütununu metnin uzunluğu ile değiştirme
dataset_numeric['review_length'] = dataset['review'].apply(len)

# Sütun adlarını metin olarak değiştirme
dataset_numeric.columns = [str(col) for col in dataset_numeric.columns]

# Standardizasyon
scaler = StandardScaler()
dataset_standardized = pd.DataFrame(scaler.fit_transform(dataset_numeric), columns=dataset_numeric.columns)

# Histogramlar, Kutu Grafikleri, Violin Grafikleri ve Scatter Plotlar
for column in dataset_numeric.columns:
    # Histogram
    plt.hist(dataset[column], bins=30, color='skyblue', edgecolor='black')
    plt.xlabel(column)
    plt.ylabel('Frekans')
    plt.title(f'{column} Değişkeninin Histogramı')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    # Kutu Grafiği
    plt.boxplot(dataset[column])
    plt.title(f'{column} Değişkeninin Kutu Grafiği')
    plt.show()

    # Violin Grafiği
    sns.violinplot(y=dataset[column], color='skyblue')
    plt.title(f'{column} Değişkeninin Violin Grafiği')
    plt.show()

    # Scatter Plot
    sns.scatterplot(data=dataset, x=column, y='rating', color='skyblue')
    plt.title(f'{column} Değişkeni ve Rating Arasındaki Scatter Plot')
    plt.xlabel(column)
    plt.ylabel('Rating')
    plt.grid()
    plt.show()

# Sayısal giriş ve çıkışlar için normalize edilmiş mutlak çapraz korelasyon haritasını hesapla ve görselleştir
correlation_matrix = dataset_standardized.corr()
sns.heatmap(np.abs(correlation_matrix), annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title('Normalize Edilmiş Mutlak Çapraz Korelasyon Haritası')
plt.show()
