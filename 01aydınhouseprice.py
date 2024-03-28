import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from stringClassifier import classify_strings

data = pd.read_csv('aydınhouseprice_csv_file.csv', parse_dates=['Son Güncelleme Tarihi'])


# Fiyat sütununu düzgün bir şekilde biçimlendirelim (TL işaretini kaldırıp sayıyı float'a dönüştürelim)
data['Fiyat'] = data['Fiyat'].str.replace(' TL', '').str.replace('.', '').astype(float)

data['Kira Getirisi'] = data['Kira Getirisi'].str.replace(' TL', '').str.replace('.', '').astype(float)

data['Aidat'] = data['Aidat'].str.replace(' TL', '').str.replace('.', '').astype(float)





# Güncelleme Tarihi sütununu tarih veri türüne dönüştürelim
data['Son Güncelleme Tarihi'] = data["Son Güncelleme Tarihi"].values.astype("float64")

# Net M2 sütununu sayısal veri türüne dönüştürelim
""" data[' Net M2'] = data[' Net M2'].str.replace('.', '').str.replace(',', '').astype(float)
 """
# Eşya Durumu sütunundaki kategorik değerleri sayısal değerlere dönüştürelim (0: Eşyalı Değil, 1: Eşyalı)
data['Eşya Durumu'] = data['Eşya Durumu'].map({'Eşyalı Değil': 0, 'Eşyalı': 1})

data["Konut Tipi"] = classify_strings(data["Konut Tipi"])

data["Mahalle"] = classify_strings(data["Mahalle"])

data["Bulunduğu Kat"] = classify_strings(data["Bulunduğu Kat"])
data["Bina Yaşı"] = classify_strings(data["Bina Yaşı"])

data["Cephe"] = classify_strings(data["Cephe"])






# Kullanım Durumu sütunundaki kategorik değerleri sayısal değerlere dönüştürelim (0: Boş, 1: Kiracılı, 2: Mülk Sahibi)
data['Kullanım Durumu'] = data['Kullanım Durumu'].map({'Boş': 0, 'Kiracılı': 1, 'Mülk Sahibi': 2})

# Isınma Tipi ve Yakıt Tipi sütunlarını dummy değişkenlere dönüştürelim
data = pd.get_dummies(data, columns=['Isınma Tipi', 'Yakıt Tipi'])




# Yeni özellikler
# Dolar, Euro, Altın, Petrol, Beton ve Demir sütunları arasında korelasyon olabilir, bu nedenle bu sütunlar arasında
# ilişkiyi ifade eden yeni bir özellik oluşturalım
""" data['Dolar:Altın Oranı'] = data['Dolar'] / data['Altın']
data['Euro:Altın Oranı'] = data['Euro'] / data['Altın']
data['Petrol:Dolar Oranı'] = data['Petrol'] / data['Dolar']
data['Petrol:Euro Oranı'] = data['Petrol'] / data['Euro']
data['Beton:Dolar Oranı'] = data['Beton'] / data['Dolar']
data['Demir:Dolar Oranı'] = data['Demir'] / data['Dolar'] """





# Giriş ve çıkış sütunlarını ayıralım
X = data.drop(columns=['Fiyat'])
y = data['Fiyat']

# Standart ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veri setinin ilk birkaç satırını kontrol edelim
print(data.head())

# Veri setindeki sütunların istatistiksel bilgilerini alalım
print(data.describe())

# Histogramlar
data.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()

# Box plotlar
plt.figure(figsize=(15, 10))
sns.boxplot(data=data)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Violin plotlar
plt.figure(figsize=(15, 10))
sns.violinplot(data=data)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Korelasyon matrisi
corr_matrix = data.corr()

# Korelasyon matrisini görselleştirelim
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Korelasyon Matrisi')
plt.tight_layout()
plt.show()
