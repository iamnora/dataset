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
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi yükle
adult_census = pd.read_csv("common_dataset_touch_features_offset.csv")

# Seçilen sütunlar
selected_columns = ["10", "100", "200", "300", "400", "500"]

# Hedef değişkenler
target_columns = ["touch_type", "touch", "finger", "palm", "fist"]

# Önişleme
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

# Her hedef değişken için model oluştur
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC()
}

# Her hedef değişken için ayrı ayrı modelleme yap
for target_col in target_columns:
    target = adult_census[target_col]
    data = adult_census[selected_columns]

    preprocessor = ColumnTransformer([
        ('one-hot-encoder', categorical_preprocessor, []),
        ('standard-scaler', numerical_preprocessor, selected_columns)])

    best_model = None
    best_score = 0

    for model_name, model in models.items():
        pipeline = make_pipeline(preprocessor, model)
        
        # Çapraz doğrulama
        cv_results = cross_validate(pipeline, data, target, cv=StratifiedKFold(n_splits=10))
        scores = cv_results["test_score"]
        mean_accuracy = scores.mean()
        std_accuracy = scores.std()
        
        print(f"{target_col} - {model_name} Cross Validation Accuracy: {mean_accuracy:.3f} +/- {std_accuracy:.3f}")
        
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
    print(f"{target_col} - Train Accuracy:", train_accuracy)

    # Test seti performans metrikleri
    test_predictions = best_model.predict(data_test)
    test_accuracy = accuracy_score(target_test, test_predictions)
    print(f"{target_col} - Test Accuracy:", test_accuracy)

    # En iyi modeli kaydet
    joblib.dump(best_model, f"Dataset14_BestModel_{target_col}")

    # Veri öncesi histogramlar
    plt.figure(figsize=(12, 6))
    for i, column in enumerate(data.columns):
        plt.subplot(2, 3, i + 1)
        sns.histplot(data[column], kde=True)
        plt.title(column)
    plt.tight_layout()
    plt.show()
