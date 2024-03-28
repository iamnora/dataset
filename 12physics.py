import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from stringClassifier import classify_strings

data = pd.read_csv('Dataset-ML.csv')


data["Öğrenci No"] = classify_strings(data["Öğrenci No"])


def turnToFloat(n):
    return n


for item in data:
    try: 
     data[item] = data[item].astype(float)
    
    except ValueError:
     data[item] = classify_strings(data[item])

  
 




# Giriş ve çıkış sütunlarını ayıralım
X = data.drop(columns=['pass'])
y = data['pass'] 
#There were no prediction outputs so we used this field as an example

# Veriyi ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

# Veri setini tablo haline getirme
df_tabular = pd.DataFrame(X_scaled, columns=X.columns)
df_tabular['Fiyat'] = y_scaled.flatten()

# Veri setinin ilk birkaç satırını kontrol edelim
print(data.head())

# Veri setindeki sütunların istatistiksel bilgilerini alalım
print(data.describe())

# Histogramlar
data.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()
num_cols = len(df_tabular.columns)

# Box plotlar
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
    sns.scatterplot(x=df_tabular[col], y=df_tabular['Fiyat'], color='purple')
    plt.title(col + ' vs Fiyat')
plt.tight_layout()
plt.show()

# Korelasyon matrisi
corr_matrix = df_tabular.corr().abs()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korelasyon Matrisi")
plt.show()
