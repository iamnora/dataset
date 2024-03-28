import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

# Veri setini yükleme
data = pd.read_csv('sh_car_price.csv')

# Giriş ve çıkış sütunlarını ayırma
X = data.drop(columns=['output_column'])
y = data['output_column']

# Sayısal olmayan sütunları filtreleme
non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns

# Sayısal olmayan sütunları veri çerçevesinden çıkarma
X_numeric = X.drop(columns=non_numeric_columns)

# Standart ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Yeni özelliklerin çıkarılması (örneğin, temel bileşen analizi - PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Veri setinin görselleştirilmesi

# Histogramlar
X_numeric.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()

# Box plotlar
plt.figure(figsize=(15, 10))
sns.boxplot(data=X_numeric)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Violin plotlar
plt.figure(figsize=(15, 10))
sns.violinplot(data=X_numeric)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatter plot (girişlerin çıkışa göre davranışı)
for col in X_numeric.columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(X_numeric[col], y)
    plt.xlabel(col)
    plt.ylabel('output_column')
    plt.title(f'{col} vs output_column')
    plt.tight_layout()
    plt.show()

# Korelasyon matrisi
corr_matrix = X_numeric.corr()

# Korelasyon matrisinin görselleştirilmesi
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Korelasyon Matrisi')
plt.tight_layout()
plt.show()

# Normalize edilmiş mutlak çapraz korelasyon haritası
plt.figure(figsize=(10, 8))
sns.heatmap(np.abs(corr_matrix), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Normalize Edilmiş Mutlak Çapraz Korelasyon Haritası')
plt.tight_layout()
plt.show()
