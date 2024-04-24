import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the data
data = pd.read_csv('humanmotion.csv')

# Data Cleaning: Convert columns to numeric and handle errors
numeric_columns = ['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_z']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop any rows with NaN values that were introduced by conversion errors
data = data.dropna()

# Preprocess the data
# Encode the target variable
label_encoder = LabelEncoder()
data['Output'] = label_encoder.fit_transform(data['Output'])

# Scale the feature variables
scaler = StandardScaler()
feature_columns = [col for col in data.columns if col != 'Output']
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# Split the data into features and target
X = data[feature_columns]
y = data['Output']

# Split the dataset using the specified parameters
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=47, stratify=y
)

# Model training using Stratified 10-fold Cross-Validation
model = RandomForestClassifier()
stratified_kfold = StratifiedKFold(n_splits=10)
cv_scores = cross_val_score(model, X_train, y_train, cv=stratified_kfold, scoring='accuracy')
print('Cross-Validation Scores:', cv_scores)
print('Average CV Accuracy:', cv_scores.mean())

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Test Accuracy:', accuracy)

# Save the model
joblib.dump(model, 'human_motion_classifier.pkl')
print('Model saved as human_motion_classifier.pkl')