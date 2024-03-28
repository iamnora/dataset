import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Önceden hazırlanmış ASL veri setini yükleyin
df = pd.read_csv("sign_language_keypoints_csv.csv")  # Veri setinizin adını ve yolunu doğru şekilde belirtin

# Veri setinin özeti
print("Veri setinin bilgileri:")
print(df.info())
print("hello")

# Özellikler ve hedef değişken arasında ilişkiyi görselleştirme
sns.pairplot(df, hue='Harf', diag_kind='hist')
plt.show()

# Veri setindeki her sütunun histogramını çizin
for column in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=True)
    plt.title(f"{column} Histogram")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

# Box plot çizimi
for column in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='letter', y=column, data=df)
    plt.title(f"{column} vs Letter Box Plot")
    plt.xlabel("Letter")
    plt.ylabel(column)
    plt.show()

# Violin plot çizimi
for column in df.columns:
    plt.figure(figsize=(8, 5))
    sns.violinplot(x='letter', y=column, data=df)
    plt.title(f"{column} vs Letter Violin Plot")
    plt.xlabel("Letter")
    plt.ylabel(column)
    plt.show()

# Özellikleri ve hedef değişkeni ayırma
X = df.drop(columns=["letter"])
y = df["letter"]

# Verileri standardize etme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalleştirilmiş mutlak çapraz korelasyon haritası oluşturma
corr = np.abs(np.corrcoef(X_scaled.T))
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=X.columns, yticklabels=X.columns)
plt.title("Normalized Absolute Cross-Correlation Map")
plt.show()
