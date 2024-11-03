import pandas as pd
import streamlit as st

data = pd.read_csv('https://raw.githubusercontent.com/kmrhrsid/jie43203/refs/heads/main/Crop_recommendation.csv')

st.write(data)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Identify Missing Values
missing_values = data.isnull().sum()

# 2. Identify Outliers using Z-score method
def identify_outliers(data, threshold=3):
    outliers = {}
    for col in data.select_dtypes(include=[np.number]).columns:
        z_scores = (data[col] - data[col].mean()) / data[col].std()
        outliers[col] = data[np.abs(z_scores) > threshold]
    return outliers

outliers = identify_outliers(data)

# 3. Identify Inconsistencies (example: inconsistent categorical data)
def identify_inconsistencies(data, column, expected_values):
    return data[~data[column].isin(expected_values)]

expected_labels = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
    'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
    'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange',
    'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']  # replace with actual expected labels
inconsistencies = identify_inconsistencies(data, 'label', expected_labels)

# 4. Identify Inaccuracies (example: negative values in a column where only positive values are expected)
def identify_inaccuracies(data, column, condition):
    return data[~data[column].apply(condition)]

inaccuracies_ph = identify_inaccuracies(data, 'ph', lambda x: 0 <= x <= 14)
inaccuracies_temperature = identify_inaccuracies(data, 'temperature', lambda x: -50 <= x <= 60)  # example temperature range
inaccuracies_humidity = identify_inaccuracies(data, 'humidity', lambda x: 0 <= x <= 100)
inaccuracies_NPK = identify_inaccuracies(data, 'N', lambda x: x >= 0) | identify_inaccuracies(data, 'P', lambda x: x >= 0) | identify_inaccuracies(data, 'K', lambda x: x >= 0)


# Display Results
print("Missing Values:\n", missing_values)
print("\nOutliers:\n", {col: len(outliers[col]) for col in outliers})
print("\nInconsistencies in 'label' Column:\n", inconsistencies)
print("\nInaccuracies in 'ph' Column:\n", inaccuracies_ph)
print("\nInaccuracies in 'temperature' Column:\n", inaccuracies_temperature)
print("\nInaccuracies in 'humidity' Column:\n", inaccuracies_humidity)
print("\nInaccuracies in 'N', 'P', 'K' Columns:\n", inaccuracies_NPK)


import pandas as pd
import numpy as np

# Assuming `data` is your DataFrame containing the data

def remove_outliers(data, outliers_counts):
    for col, count in outliers_counts.items():
        if count > 0:
            z_scores = (data[col] - data[col].mean()) / data[col].std()
            data = data[np.abs(z_scores) <= 1]  # Adjusted threshold to 1 for z-score
    return data

# Given outliers counts
outliers_counts = {'N': 0, 'P': 0, 'K': 94, 'temperature': 33, 'humidity': 0, 'ph': 30, 'rainfall': 22}

# Remove outliers
new_data = remove_outliers(data, outliers_counts)

# Check if there are still outliers
outliers_remaining = {col: len(new_data[new_data[col].isna()]) for col in new_data.columns}

if any(outliers_remaining[col] > 0 for col in outliers_remaining):
    print("After removing outliers, there are still outliers remaining.")
else:
    print("After removing outliers, there are no outliers remaining.")


round_to_3_decimals = lambda x: round(x, 3) if isinstance(x, float) else x

df = new_data.applymap(round_to_3_decimals)

df


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Histogram for 'N'
plt.figure(figsize=(10, 6))
sns.histplot(df['N'], bins=30, kde=True)
plt.title('Distribution of Nitrogen Content (N)')
plt.xlabel('N')
plt.ylabel('Frequency')
plt.show()

# Boxplot for 'temperature'
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['temperature'])
plt.title('Boxplot of Temperature')
plt.xlabel('Temperature')
plt.show()

# Bar chart for 'label'
plt.figure(figsize=(12, 8))
sns.countplot(x=df['label'])
plt.title('Frequency of Each Label')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pairplot of selected numerical features
plt.figure(figsize=(12, 10))
sns.pairplot(df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']])
plt.show()

# Heatmap of correlation matrix
plt.figure(figsize=(12, 8))
corr_matrix = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import OneClassSVM

# Generate synthetic data for clustering
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Fit One-Class SVM
svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)  # Adjust parameters as needed
svm.fit(X)

# Predict outliers/anomalies
y_pred = svm.predict(X)

# Plotting the results
plt.figure(figsize=(8, 6))

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, edgecolors='k')

plt.title('One-Class SVM for Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()


st.pyplot(plt.gcf())
