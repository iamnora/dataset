import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from stringClassifier import classify_strings

adult_census = pd.read_csv("accident_news.csv", encoding="latin-1", delimiter=",")

selected_columns = ["Content", "Province_Code", "otomobil", "motosiklet"]

data = adult_census[selected_columns]

data["Content"] = classify_strings(data["Content"])
data["Province_Code"] = classify_strings(data["Province_Code"])

print(data.isnull().sum())
data.dropna(inplace=True)

target_names = ["Date", "Location"]  # List of target column names

adult_census["Date"] = pd.to_datetime(adult_census["Date"]).astype('int64') // 10**9  # Convert date to timestamp in seconds

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, ["Content", "Province_Code"]),
    ('standard-scaler', numerical_preprocessor, ["otomobil", "motosiklet"])])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Support Vector Machine": SVR()
}

for target_name in target_names:
    target = adult_census[target_name]
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=47)
    
    best_model = None
    best_score = float('-inf')  # Negative infinity

    for model_name, model in models.items():
        pipeline = make_pipeline(preprocessor, model)
        
        cv_results = cross_validate(pipeline, data_train, target_train, cv=KFold(n_splits=10), scoring='r2')
        scores = cv_results["test_score"]
        mean_r2 = scores.mean()
        std_r2 = scores.std()
        
        print(f"{target_name} - {model_name} Cross Validation R2 Score: {mean_r2:.3f} +/- {std_r2:.3f}")
        
        if mean_r2 > best_score:
            best_model = pipeline
            best_score = mean_r2

    if best_model:
        best_model.fit(data_train, target_train)

        train_predictions = best_model.predict(data_train)
        train_r2 = r2_score(target_train, train_predictions)
        print(f"{target_name} - Train R2 Score:", train_r2)

        test_predictions = best_model.predict(data_test)
        test_r2 = r2_score(target_test, test_predictions)
        print(f"{target_name} - Test R2 Score:", test_r2)

        joblib.dump(best_model, f"Dataset11_BestModel_{target_name}.joblib")
