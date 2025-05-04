import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import os

# Create output directory for plots
os.makedirs('../Visualizations', exist_ok=True)

# Load the data with the correct file path
df = pd.read_csv(r'C:\Users\Addison Thompson\Downloads\Heart_Predictions_Dataset\Heart Prediction Quantum Dataset.csv')

# Clean column names
df.columns = df.columns.str.strip('`')

# Basic EDA
print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Create correlation matrix
plt.figure(figsize=(10, 8))
# Create a custom subtle slate grey and orange diverging palette
colors = sns.diverging_palette(260, 30, s=30, l=65, n=200, center="light")
sns.heatmap(df.corr(), annot=True, cmap=colors, center=0, vmin=-1, vmax=1,
            annot_kws={'size': 10}, fmt='.2f', square=True)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('../Visualizations/correlation_matrix.png')
plt.close()

# Distribution of features
features = ['Age', 'BloodPressure', 'Cholesterol', 'HeartRate', 'QuantumPatternFeature']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, feature in enumerate(features):
    sns.histplot(data=df, x=feature, hue='HeartDisease', bins=30, ax=axes[idx])
    axes[idx].set_title(f'{feature} Distribution')

plt.tight_layout()
plt.savefig('../Visualizations/feature_distributions.png')
plt.close()

# Box plots for each feature
plt.figure(figsize=(15, 6))
df.boxplot(column=features)
plt.title('Feature Distributions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../Visualizations/feature_boxplots.png')
plt.close()

# Prepare data for modeling
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a simple logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Print model performance
print("\nModel Performance:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('../Visualizations/feature_importance.png')
plt.close()

print("\nAnalysis complete! Check the 'Visualizations' directory for visualizations.") 