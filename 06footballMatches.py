from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import seaborn as sns

from sklearn.pipeline import make_pipeline

# Veriyi yükle
adult_census = pd.read_csv("footballmathces_v2.csv")

# Sadece istediğiniz sütunları seçin
selected_columns = [ 'home_team_data_squad_size', 'home_team_data_team_value', 'home_team_data_team_','away_team_data_squad_size', 'away_team_data_team_value', 'away_team_data_team_trophies']
data = adult_census[selected_columns]


# Hedef değişken
target_name = "Result"
target = adult_census[target_name]

# Önişleme
numerical_preprocessor = StandardScaler()

preprocessor = ColumnTransformer([

    ('standard-scaler', numerical_preprocessor, selected_columns)])

# Modeller
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC()
}


# Model seçimi ve değerlendirme
best_model = None
best_score = 0

for model_name, model in models.items():
    pipeline = make_pipeline(preprocessor, model)
    
    # Çapraz doğrulama
    cv_results = cross_validate(pipeline, data, target, cv=StratifiedKFold(n_splits=10))
    scores = cv_results["test_score"]
    mean_accuracy = scores.mean()
    std_accuracy = scores.std()
    
    print(f"{model_name} Cross Validation Accuracy: {mean_accuracy:.3f} +/- {std_accuracy:.3f}")
    
    # En iyi modeli seç
    if mean_accuracy > best_score:
        best_model = pipeline
        best_score = mean_accuracy

# Train-test ayırma
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=47, stratify=target)

# En iyi modelle eğitim
best_model.fit(data_train, target_train)

# Eğitim seti performans metrikleri
train_predictions = best_model.predict(data_train)
train_accuracy = accuracy_score(target_train, train_predictions)
print("Train Accuracy:", train_accuracy)

# Test seti performans metrikleri
test_predictions = best_model.predict(data_test)
test_accuracy = accuracy_score(target_test, test_predictions)
print("Test Accuracy:", test_accuracy)

# En iyi modeli kaydet
joblib.dump(best_model, "football_Model.pkl")








# Eğitim verisi histogramları
plt.figure(figsize=(12, 6))
for i, column in enumerate(data_train.columns):
    plt.subplot(2, 3, i + 1)
    sns.histplot(data_train[column], kde=True)
    plt.title(column)
plt.tight_layout()
plt.show()



# Eğitim verisi violin plotları
plt.figure(figsize=(12, 6))
for i, column in enumerate(data_train.columns):
    plt.subplot(2, 3, i + 1)
    sns.violinplot(x=target_train, y=data_train[column])
    plt.title(column)
plt.tight_layout()
plt.show()


