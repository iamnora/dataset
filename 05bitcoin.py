import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import joblib


# Load the data
data = pd.read_csv('bitcoindataset.csv')

# Encode the categorical 'Command' column
encoder = LabelEncoder()
data['Command_Encoded'] = encoder.fit_transform(data['Command'])
data.drop('Command', axis=1, inplace=True)

# Set up features and target
X = data.drop('Close_Predictions', axis=1)
y = data['Close_Predictions']

# Define models for cross-validation
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=47),
    "SVR": SVR()
}

# Perform K-Fold cross-validation
cv = KFold(n_splits=10)
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, scoring='r2', cv=cv)
    results[name] = scores.mean()
print("Cross-validation results:", results)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47, stratify=None)

# Train the best model (Linear Regression in this case)
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
train_predictions = model.predict(X_train)
train_r2 = r2_score(y_train, train_predictions)
test_predictions = model.predict(X_test)
test_r2 = r2_score(y_test, test_predictions)
print("Train R² score:", train_r2)
print("Test R² score:", test_r2)

# Save the model
model_filename = 'linear_regression_model.joblib'
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")
