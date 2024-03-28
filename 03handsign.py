import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('sign_language_keypoints_csv.csv', sep=';')
dataset.drop(columns=['Harf'], inplace=True)


features = []
for i in range(1, 22):
    x_col = f'X{i}'
    y_col = f'Y{i}'
    features.extend([x_col, y_col])

feature_data = {}
for feature in features:
    feature_data[f'Mean_{feature}'] = [dataset[feature].mean()]
    feature_data[f'Std_{feature}'] = [dataset[feature].std()]
    feature_data[f'Var_{feature}'] = [dataset[feature].var()]
    feature_data[f'Min_{feature}'] = [dataset[feature].min()]
    feature_data[f'Max_{feature}'] = [dataset[feature].max()]
    feature_data[f'Mode_{feature}'] = [dataset[feature].mode().values[0]]
    feature_data[f'Median_{feature}'] = [dataset[feature].median()]
    feature_data[f'Quantile1_{feature}'] = [dataset[feature].quantile(0.25)]
    feature_data[f'Quantile3_{feature}'] = [dataset[feature].quantile(0.75)]
    feature_data[f'Skew_{feature}'] = [dataset[feature].skew()]
    feature_data[f'Kurt_{feature}'] = [dataset[feature].kurt()]

feature_df = pd.DataFrame(feature_data)

feature_df = pd.DataFrame(feature_data)

feature_df.to_csv('Feature_Extraction_Results.csv', index=False)

correlation_matrix = dataset.corr()
correlation_matrix.to_csv('Correlation_Matrix.csv')

sns.heatmap(np.abs(correlation_matrix), annot=True, cmap='coolwarm', vmin=0, vmax=1, fmt='.2f')
plt.title('Normalized Absolute Cross-Correlation Map')
plt.show()

# Histogram
for column in dataset.columns:
    plt.hist(dataset[column], bins=30, color='skyblue', edgecolor='black')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column}')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# Scatter plot
for i in range(len(dataset.columns)):
    for j in range(i + 1, len(dataset.columns)):
        plt.scatter(dataset.iloc[:, i], dataset.iloc[:, j], color='skyblue', edgecolor='black', s=10)
        plt.title(f'Scatter Plot of {dataset.columns[i]}-{dataset.columns[j]}')
        plt.grid(alpha=0.75)
        plt.show()

# Violin plot
for column in dataset.columns:
    sns.violinplot(y=dataset[column], color='skyblue')
    plt.title(f'Violin Plot of {column}')
    plt.show()

# Box plot
for column in dataset.columns:
    plt.boxplot(dataset[column])
    plt.title(f'Box Plot of {column}')
    plt.show()

scaler = StandardScaler()
dataset_standardized = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)
dataset_standardized.to_csv('Standardized_Columns.csv', index=False)
