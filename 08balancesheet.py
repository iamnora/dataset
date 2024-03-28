import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
dataset = pd.read_csv('BalanceSheetDataSet.csv')

# Check for missing values
print("Missing values before preprocessing:")
print(dataset.isnull().sum())

# Drop rows with missing values
dataset.dropna(inplace=True)

# Check for duplicate rows
print("Duplicate rows before preprocessing:", dataset.duplicated().sum())

# Remove duplicate rows
dataset.drop_duplicates(inplace=True)

# Separate inputs and outputs
X = dataset.drop(columns=['Dönem Net Kar/Zararı'])
y = dataset['Dönem Net Kar/Zararı']

# Standardize inputs
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Standardize outputs
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Convert scaled arrays back to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
y_scaled_df = pd.DataFrame(y_scaled, columns=['Dönem Net Kar/Zararı'])

# Feature engineering: Create a new feature by multiplying 'Nakit ve Nakit Benzerleri' with 'Finansal Yatırımlar'
X_scaled_df['New_Feature'] = X_scaled_df['Nakit ve Nakit Benzerleri'] * X_scaled_df['Finansal Yatırımlar']

# Visualization
for column in X_scaled_df.columns:
    sns.histplot(X_scaled_df[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.show()

for column in X_scaled_df.columns:
    sns.barplot(x=X_scaled_df[column], y=y_scaled_df['Dönem Net Kar/Zararı'])
    plt.title(f'Bar chart of {column} vs Dönem Net Kar/Zararı')
    plt.show()

for column in X_scaled_df.columns:
    sns.scatterplot(x=X_scaled_df[column], y=y_scaled_df['Dönem Net Kar/Zararı'])
    plt.title(f'Scatter plot of {column} vs Dönem Net Kar/Zararı')
    plt.show()

for column in X_scaled_df.columns:
    sns.violinplot(x=X_scaled_df[column])
    plt.title(f'Violin plot of {column}')
    plt.show()

print("Summary statistics of inputs:")
print(X_scaled_df.describe())
print("Summary statistics of output:")
print(y_scaled_df.describe())

# Calculate correlation matrix
correlation_matrix = X_scaled_df.corr().abs()

# Plot normalized absolute cross-correlation map using seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Normalized Absolute Cross-Correlation Map of Inputs')
plt.show()

# Calculate correlation between inputs and output
correlation_with_output = X_scaled_df.corrwith(y_scaled_df['Dönem Net Kar/Zararı'])

# Plot normalized absolute cross-correlation map between inputs and output
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_with_output.to_frame(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Normalized Absolute Cross-Correlation Map between Inputs and Output')
plt.show()
