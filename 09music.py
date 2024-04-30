import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from stringClassifier import classify_strings

# Veriyi yükle
adult_census = pd.read_csv("Music Informations and Lyrics_ from Spotify and Musixmatch.csv")

# Sadece istediğiniz sütunları seçin
selected_columns = ["Artist","Artist Popularity","Artist Genres","Track Duration(ms)","Danceability","Energy","Key","Loudness","Acousticness","Liveness","Valence","Tempo"]
data = adult_census[selected_columns]

data["Artist"] = classify_strings(data["Artist"])
data["Artist Genres"] = classify_strings(data["Artist Genres"])


# Hedef değişken
target_name = "Track Popularity"
target = adult_census[target_name]

# Önişleme
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, selector(dtype_include="object")),
    ('standard-scaler', numerical_preprocessor, selector(dtype_exclude="object"))
])

# Modeli oluştur
model = make_pipeline(preprocessor, LinearRegression())



# Veriyi eğitim ve test setlerine ayır
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=47)

# Modeli eğit
_ = model.fit(data_train, target_train)

# Eğitim seti üzerinde tahminler yap
train_predictions = model.predict(data_train)

# Eğitim seti performansını ölç
train_r2 = r2_score(target_train, train_predictions)

# Test seti üzerinde tahminler yap
test_predictions = model.predict(data_test)

# Test seti performansını ölç
test_r2 = r2_score(target_test, test_predictions)

print("Eğitim R^2 skoru:", train_r2)
print("Test R^2 skoru:", test_r2)

# Modeli kaydet
joblib.dump(model, "musicModel.pkl")

# Eğitim verisi histogramları
plt.figure(figsize=(12, 6))
for i, column in enumerate(data_train.columns):
    plt.subplot(4, 4, i + 1)
    sns.histplot(data_train[column], kde=True)
    plt.title(column)
plt.tight_layout()
plt.show()

# Eğitim verisi scatter plotları
plt.figure(figsize=(12, 6))
for i, column in enumerate(data_train.columns):
    plt.subplot(4, 4, i + 1)
    sns.scatterplot(x=target_train, y=data_train[column])
    plt.title(column)
plt.tight_layout()
plt.show()

# Eğitim verisi violin plotları
plt.figure(figsize=(12, 6))
for i, column in enumerate(data_train.columns):
    plt.subplot(4, 4, i + 1)
    sns.violinplot(x=target_train, y=data_train[column])
    plt.title(column)
plt.tight_layout()
plt.show()


