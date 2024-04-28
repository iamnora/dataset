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
from stringClassifier import classify_strings
from sklearn.pipeline import make_pipeline

# Veriyi yükle
adult_census = pd.read_csv("accident_news.csv")

# Sadece istediğiniz sütunları seçin
selected_columns = ["Content", "Location", "Province_Code", "Vehicles", "Death"]
data = adult_census[selected_columns]

# Sınıflandırma için metin sütunlarını işle
data["Content"] = classify_strings(data["Content"])
data["Location"] = classify_strings(data["Location"])
data["Province_Code"] = classify_strings(data["Province_Code"])
data["Vehicles"] = classify_strings(data["Vehicles"])

# Hedef değişken
target_name = "Injured"
target = adult_census[target_name]

# Önişleme
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, ["Content", "Location", "Province_Code"]),
    ('standard-scaler', numerical_preprocessor, ["Vehicles", "Death"])])

# Modeller
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC()
}

# Performans metrikleri
metrics = {}

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
joblib.dump(best_model, "best_model.pkl")
