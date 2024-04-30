import pandas as pd
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


adult_census = pd.read_csv("arcgis_ngas_usak_v02.csv")

target_name = "Total"
target = adult_census[target_name]

X = adult_census.drop(columns=[target_name])

numerical_preprocessor = StandardScaler()
preprocessor = ColumnTransformer([
    ('standard-scaler', numerical_preprocessor, ["Year", "Jan usd/tr", "May usd/tr"])
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Support Vector Regressor": SVR()
}

best_model = None
best_score = float('-inf')

for model_name, model in models.items():
    pipeline = make_pipeline(preprocessor, model)
    
    # k-fold cross-validation
    cv_results = cross_val_score(pipeline, X, target, cv=ShuffleSplit(n_splits=10, test_size=0.2))
    mean_r2 = cv_results.mean()
    std_r2 = cv_results.std()
    
    print(f"{model_name} Cross Validation R^2 Score: {mean_r2:.3f} +/- {std_r2:.3f}")
    
    if mean_r2 > best_score:
        best_model = pipeline
        best_score = mean_r2


plt.figure(figsize=(12, 6))
for i, column in enumerate(target.columns):
    plt.subplot(2, 3, i + 1)
    sns.histplot(target[column], kde=True)
    plt.title(column)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
for i, column in enumerate(target.columns):
    plt.subplot(2, 3, i + 1)
    sns.violinplot(y=target[column])
    plt.title(column)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
for i, column in enumerate(target.columns):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(y=target[column])
    plt.title(column)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
for i, column in enumerate(target.columns):
    plt.subplot(2, 3, i + 1)
    plt.scatter(target[column], target, alpha=0.5)
    plt.xlabel(column)
    plt.ylabel("Total")
plt.tight_layout()
plt.show()


joblib.dump(best_model, "Dataset15_BestModel.pkl")


X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.2, random_state=47)


best_model.fit(X_train, target_train)


train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)
train_r2 = r2_score(target_train, train_predictions)
test_r2 = r2_score(target_test, test_predictions)
print("Train R^2 Score:", train_r2)
print("Test R^2 Score:", test_r2)
