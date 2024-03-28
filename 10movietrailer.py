import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Veri setini yükle
df = pd.read_csv('moviedata.csv')

# Sıfır boyutlu dizileri atlayarak sadece 1 boyutlu veya daha büyük dizileri alın
filtered_red_hist = [x for x in df['red_hist'].values if len(x) > 0]

# Her bir NumPy dizisini düzleştirin ve listeye ekleyin
flat_red_hist = np.hstack(filtered_red_hist)

# Dizeyi uygun şekilde işleyerek float'a dönüştürme işlemi
cleaned_red_hist = filtered_red_hist[0].replace("[", "").replace("]", "")
flat_red_hist = np.array([float(val) for val in cleaned_red_hist.split()])

# Sıfır boyutlu dizileri atlayarak sadece 1 boyutlu veya daha büyük dizileri alın
filtered_band_last = [x for x in df['band_last'].values if pd.notnull(x) and isinstance(x, str)]

if filtered_band_last:
    flat_band_last = np.hstack(filtered_band_last)
else:
    # Burada hata mesajını ayarlayabilir veya gerekirse başka bir işlem yapabilirsiniz
    print("filtered_band_last listesi boş!")

# Dizeyi uygun şekilde işleyerek float'a dönüştürme işlemi
cleaned_band_last = filtered_band_last[0].replace("[", "").replace("]", "") if filtered_band_last else ''
flat_band_last = np.array([float(val) for val in cleaned_band_last.split()]) if cleaned_band_last else np.array([])

# Veri setinin ilk birkaç satırını görüntüle
print(df.head())

# Veri setinin betimsel istatistiklerini görüntüle
print(df.describe())

# Veri setindeki eksik değerleri kontrol et
print(df.isnull().sum())

# Standardize the input features
scaler_rgb = StandardScaler()
scaler_sound = StandardScaler()

rgb_histograms_scaled = scaler_rgb.fit_transform(flat_red_hist.reshape(-1, 1))

# Eğer işlem tamamsa, düzleştirilmiş diziyi orijinal şekline dönüştürün
if rgb_histograms_scaled is not None:
    rgb_histograms_scaled = rgb_histograms_scaled.reshape(-1, 1)

sound_features_scaled = scaler_sound.fit_transform(flat_band_last.reshape(-1,1))

# Extract new features if possible

# Create a tabular dataset
data = np.concatenate((rgb_histograms_scaled, sound_features_scaled), axis=1)
columns = ['rate', 'red_hist', 'green_hist', 'blue_hist', 'std_pow', 'max_pow', 'min_pow', 'mean_pow', 'max_pow_freq', 'max_pow_time', 'band5', 'band10', 'band15', 'band20', 'band25', 'band30', 'band35', 'band40', 'band45', 'band50', 'band55', 'band60', 'band65', 'band70', 'band75', 'band80', 'band85', 'band90', 'band95', 'band100', 'band105', 'band110', 'band115', 'band120', 'band_last']
df = pd.DataFrame(data, columns=columns)

# Statistical information
stats_info = df.describe()

# Plot histograms, box plots, and violin plots
plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns):
    plt.subplot(3, 4, i+1)
    sns.histplot(df[column], kde=True)
    plt.title(column)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns):
    plt.subplot(3, 4, i+1)
    sns.boxplot(y=df[column])
    plt.title(column)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns):
    plt.subplot(3, 4, i+1)
    sns.violinplot(y=df[column])
    plt.title(column)
plt.tight_layout()
plt.show()

# Scatter plot or bar plot for behavior of each input according to its output
plt.figure(figsize=(15, 10))
for i, column in enumerate(columns):
    plt.subplot(3, 4, i+1)
    sns.scatterplot(data=df, x=column, y='IMDB_score')  # 'IMDB_score' sütununu uygun bir şekilde değiştirin
    plt.title(column)
plt.tight_layout()
plt.show()

# Normalized absolute cross-correlation map
corr_matrix = df.corr().abs()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Absolute Cross-correlation Map")
plt.show()
