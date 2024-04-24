import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
# Assuming you have a DataFrame named df

# Preprocessing
# Standardization of inputs
from sklearn.preprocessing import StandardScaler
import pandas as pd



df = pd.read_csv('common_dataset_touch_features_offset.csv')


number_strings = []

for i in range(1, 3201):
    number_strings.append(str(i))


input_columns = number_strings  # List of input column names
output_columns = ["user_id","touch_type","touch","finger","palm","fist"]  # List of output column names

scaler = StandardScaler()
df[input_columns] = scaler.fit_transform(df[input_columns])

# Show effects of preprocessing
print(df.head())  # Check the first few rows to see the standardized values

# Feature engineering
# You may perform feature extraction here if needed

# Plotting
# Histograms
df[input_columns].hist(figsize=(10, 10))
plt.show()

# Bar charts
for col in output_columns:
    sns.countplot(x=col, data=df)
    plt.show()

# Scatter plots
for col in input_columns:
    for output_col in output_columns:
        sns.scatterplot(x=col, y=output_col, data=df)
        plt.show()

# Violin plots
for col in output_columns:
    sns.violinplot(x=col, y='user_id', data=df)
    plt.show()

# Statistical information
print(df.describe())

# Histogram, box plot, and violin plot of each column
for col in df.columns:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f"Histogram of {col}")

    plt.subplot(1, 3, 2)
    sns.boxplot(y=df[col])
    plt.title(f"Box Plot of {col}")

    plt.subplot(1, 3, 3)
    sns.violinplot(y=df[col])
    plt.title(f"Violin Plot of {col}")

    plt.tight_layout()
    plt.show()

# Normalized absolute cross-correlation map
corr_matrix = df.corr().abs()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Absolute Cross-correlation Map')
plt.show()
