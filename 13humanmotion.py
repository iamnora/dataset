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

adult_census = pd.read_csv("human_motion_detection.csv", sep=';')


selected_columns = ["gyro_x", "gyro_y", "accel_z", "std_acc_30", "mean_gyro_20", "max_acc_15"]
data = adult_census[selected_columns]





target_name = "Output"
target = adult_census[target_name]


categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, []),
    ('standard-scaler', numerical_preprocessor, ["gyro_x", "gyro_y", "accel_z", "std_acc_30", "mean_gyro_20", "max_acc_15"])])


models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC()
}




best_model = None
best_score = 0

for model_name, model in models.items():
    pipeline = make_pipeline(preprocessor, model)
    

    cv_results = cross_validate(pipeline, data, target, cv=StratifiedKFold(n_splits=10))
    scores = cv_results["test_score"]
    mean_accuracy = scores.mean()
    std_accuracy = scores.std()
    
    print(f"{model_name} Cross Validation Accuracy: {mean_accuracy:.3f} +/- {std_accuracy:.3f}")
    
 
    if mean_accuracy > best_score:
        best_model = pipeline
        best_score = mean_accuracy


data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=47, stratify=target)


best_model.fit(data_train, target_train)


train_predictions = best_model.predict(data_train)
train_accuracy = accuracy_score(target_train, train_predictions)
print("Train Accuracy:", train_accuracy)


test_predictions = best_model.predict(data_test)
test_accuracy = accuracy_score(target_test, test_predictions)
print("Test Accuracy:", test_accuracy)


joblib.dump(best_model, "Dataset13_BestModel")



plt.figure(figsize=(12, 6))
for i, column in enumerate(data.columns):
    plt.subplot(2, 3, i + 1)
    sns.histplot(data[column], kde=True)
    plt.title(column)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for i, column in enumerate(data.columns):
    plt.subplot(2, 3, i + 1)
    sns.violinplot(y=data[column])
    plt.title(column)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for i, column in enumerate(data.columns):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(y=data[column])
    plt.title(column)
plt.tight_layout()
plt.show()

