import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Özellikleri içeren veri çerçevesini oluşturma
df = pd.read_csv('bitcoindataset.csv')
print(df.head())

# Özellikleri hesapla
df_features = pd.DataFrame()

# MA 20
df_features['MA_20'] = df[['Open', 'High', 'Low', 'Close']].rolling(window=20).mean().iloc[-1]

# MA 20 close
df_features['MA_20_close'] = df['Close'].rolling(window=20).mean().iloc[-1]

# STD 10 high
df_features['STD_10_high'] = df['High'].rolling(window=10).std().iloc[-1]

# PP 10 vol
df_features['PP_10_vol'] = df['Volume'].diff().abs().rolling(window=10).max().iloc[-1]

# Bağımsız ve bağımlı değişkenleri ayır
X = df.drop(['Close', 'Command'], axis=1)  # Bağımsız değişkenler
y = df['Close_Predictions']  # Bağımlı değişkenler
print("X shape:", X.shape)
print("y shape:", y.shape)

# Veriyi ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

# Veri setini tablo haline getirme
df_tabular = pd.DataFrame(X_scaled, columns=X.columns)
df_tabular['Close_Predictions'] = y_scaled.flatten()

# Histogramlar
plt.figure(figsize=(12, 6))
num_cols = len(df_tabular.columns)
for i, col in enumerate(df_tabular.columns):
    plt.subplot(2, (num_cols+1)//2, i + 1)
    sns.histplot(df_tabular[col], kde=True, color='blue', bins=10)
    plt.title(col)
plt.tight_layout()
plt.show()

# Box plot
plt.figure(figsize=(12, 6))
for i, col in enumerate(df_tabular.columns):
    plt.subplot(2, (num_cols+1)//2, i + 1)
    sns.boxplot(y=df_tabular[col], color='green')
    plt.title(col)
plt.tight_layout()
plt.show()

# Violin plot
plt.figure(figsize=(12, 6))
for i, col in enumerate(df_tabular.columns):
    plt.subplot(2, (num_cols+1)//2, i + 1)
    sns.violinplot(y=df_tabular[col], color='orange')
    plt.title(col)
plt.tight_layout()
plt.show()

# Scatter plot
plt.figure(figsize=(15, 10))
for i, col in enumerate(df_tabular.columns[:-1]):
    plt.subplot(2, (num_cols+1)//2, i + 1)
    sns.scatterplot(x=df_tabular[col], y=df_tabular['Close_Predictions'], color='purple')
    plt.title(col + ' vs Close_Predictions')
plt.tight_layout()
plt.show()

# Korelasyon matrisi
corr_matrix = df_tabular.corr().abs()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korelasyon Matrisi")
plt.show()
