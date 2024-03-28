import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv
# Veri setini yükleyin 


# CSV dosyasını oku
df = pd.read_csv('humanmotion.csv', thousands=',', decimal='.', header=0)

# İlk satırı başlık olarak atla
df = df.drop(index=0)

# Sütun adlarını belirle
df.columns = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z", "std_acc_30", "std_gyro_10", "mean_acc_20", "mean_gyro_20", "max_acc_15", "min_acc_20", "Output"]

# Veri setini görüntüle
print(df.head())

# 'Output' sütununu kaldır
X = df.drop(columns=["Output"])

# Veri setini analiz et veya işlemlere devam et


# Standartlaştırma
from sklearn.preprocessing import StandardScaler, scale

# Ölçeklendirme nesnesini tanımla
scaler = StandardScaler()

# Veriyi ölçeklendir
scaled_data = scaler.fit_transform(X)

# Ölçeklendirilmiş veriyi DataFrame'e dönüştür
df_scaled = pd.DataFrame(scaled_data, columns=X.columns)

# 'Output' sütununu ekleyin
df_scaled["Output"] = df["Output"]

# Scatterplot'u çiz
sns.scatterplot(data=df_scaled, x='gyro_x', y='gyro_y', hue='Output')


# Görselleştirmeler
# Örnek olarak, 2D scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_scaled, x='gyro_x', y='gyro_y', hue='Output')
plt.title('Scatter Plot of gyro_x vs gyro_y with Output')
plt.show()

# Histogramlar
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='gyro_x', hue='Output', multiple='stack', kde=True)
plt.title('Histogram of gyro_x with Output')
plt.show()

# Kutu grafikleri
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Output', y='gyro_x')
plt.title('Box Plot of gyro_x by Output')
plt.show()

# Keman grafikleri
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Output', y='gyro_x')
plt.title('Violin Plot of gyro_x by Output')
plt.show()

# Çapraz korelasyon haritası
numeric_df = df.drop(columns=["Output"])
corr = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

