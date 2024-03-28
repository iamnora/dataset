import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Veri setini yükle
data = pd.read_csv("usakgas.csv")

# Özellik mühendisliği için yeni özellikler ekleyebiliriz, örneğin aylık ortalama tüketim
data['Monthly_Avg_Consumption'] = data.iloc[:, 16:28].mean(axis=1)

# Toplam tüketim miktarlarını hesapla
data['Total_Consumption'] = data.iloc[:, 16:28].sum(axis=1)

# Standartlaştırma
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.iloc[:, 1:])  # İlk sütun (yıl) hariç tüm sütunları standartlaştır

# Standartlaştırılmış veriyi bir DataFrame'e dönüştür
scaled_df = pd.DataFrame(scaled_data, columns=data.columns[1:])

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=data.iloc[:, 16:], kde=True, bins=20)
plt.title("Gas Consumption Histogram")
plt.xlabel("Gas Consumption (m3)")
plt.ylabel("Frequency")
plt.show()

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=data.iloc[:, 16:])
plt.title("Gas Consumption Boxplot")
plt.xlabel("Month")
plt.ylabel("Gas Consumption (m3)")
plt.show()

# Scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data["Total_Consumption"], y=data["Monthly_Avg_Consumption"])
plt.title("Scatterplot of Total Consumption vs. Monthly Average Consumption")
plt.xlabel("Total Consumption (m3)")
plt.ylabel("Monthly Average Consumption (m3)")
plt.show()

# Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=data.iloc[:, 16:])
plt.title("Gas Consumption Violin Plot")
plt.xlabel("Month")
plt.ylabel("Gas Consumption (m3)")
plt.show()

# Korelasyon matrisi
corr_matrix = data.iloc[:, 1:].corr().abs()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korelasyon Matrisi")
plt.show()
