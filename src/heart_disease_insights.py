import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Create output directory if it doesn't exist
os.makedirs('../Visualizations/insights', exist_ok=True)

# Load the data
data_path = r'C:\Users\Addison Thompson\Downloads\Heart_Predictions_Dataset\Heart Prediction Quantum Dataset.csv'
print(f"Loading data from: {os.path.abspath(data_path)}")
df = pd.read_csv(data_path)

# Clean column names
df.columns = df.columns.str.strip('`')
print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Set style for better visualizations
plt.style.use('ggplot')
sns.set_style("whitegrid")

# 1. Age Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='Age', hue='HeartDisease', bins=30, kde=True)
plt.title('Age Distribution by Heart Disease Status')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(['No Heart Disease', 'Heart Disease'])
plt.tight_layout()
plt.savefig('../Visualizations/insights/age_distribution.png')
plt.close()

# 2. Risk Factors Analysis
plt.figure(figsize=(15, 10))

# Blood Pressure
plt.subplot(2, 2, 1)
sns.boxplot(data=df, x='HeartDisease', y='BloodPressure')
plt.title('Blood Pressure Distribution by Heart Disease Status')
plt.xlabel('Heart Disease')
plt.ylabel('Blood Pressure')

# Cholesterol
plt.subplot(2, 2, 2)
sns.boxplot(data=df, x='HeartDisease', y='Cholesterol')
plt.title('Cholesterol Distribution by Heart Disease Status')
plt.xlabel('Heart Disease')
plt.ylabel('Cholesterol')

# Heart Rate
plt.subplot(2, 2, 3)
sns.boxplot(data=df, x='HeartDisease', y='HeartRate')
plt.title('Heart Rate Distribution by Heart Disease Status')
plt.xlabel('Heart Disease')
plt.ylabel('Heart Rate')

# Quantum Pattern Feature
plt.subplot(2, 2, 4)
sns.boxplot(data=df, x='HeartDisease', y='QuantumPatternFeature')
plt.title('Quantum Pattern Feature Distribution by Heart Disease Status')
plt.xlabel('Heart Disease')
plt.ylabel('Quantum Pattern Feature')

plt.tight_layout()
plt.savefig('../Visualizations/insights/risk_factors.png')
plt.close()

# 3. Gender Analysis
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Gender', hue='HeartDisease')
plt.title('Gender Distribution by Heart Disease Status')
plt.xlabel('Gender (0=Female, 1=Male)')
plt.ylabel('Count')
plt.legend(['No Heart Disease', 'Heart Disease'])
plt.tight_layout()
plt.savefig('../Visualizations/insights/gender_analysis.png')
plt.close()

# 4. Age Group Analysis
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 40, 50, 60, 70, 100], 
                       labels=['<40', '40-50', '50-60', '60-70', '70+'])

plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='AgeGroup', hue='HeartDisease')
plt.title('Age Group Distribution by Heart Disease Status')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(['No Heart Disease', 'Heart Disease'])
plt.tight_layout()
plt.savefig('../Visualizations/insights/age_group_analysis.png')
plt.close()

# 5. Risk Factor Correlation
plt.figure(figsize=(12, 8))
sns.heatmap(df[['Age', 'BloodPressure', 'Cholesterol', 'HeartRate', 'QuantumPatternFeature', 'HeartDisease']].corr(),
            annot=True, cmap='coolwarm', center=0)
plt.title('Risk Factor Correlation Matrix')
plt.tight_layout()
plt.savefig('../Visualizations/insights/risk_factor_correlation.png')
plt.close()

# Generate statistical insights
print("\n=== Heart Disease Insights ===")

# Age-related insights
print("\nAge-related Statistics:")
print(f"Average age of patients with heart disease: {df[df['HeartDisease'] == 1]['Age'].mean():.1f} years")
print(f"Average age of patients without heart disease: {df[df['HeartDisease'] == 0]['Age'].mean():.1f} years")

# Risk factor statistics
print("\nRisk Factor Statistics:")
for factor in ['BloodPressure', 'Cholesterol', 'HeartRate', 'QuantumPatternFeature']:
    print(f"\n{factor} Statistics:")
    print(f"Average {factor} with heart disease: {df[df['HeartDisease'] == 1][factor].mean():.1f}")
    print(f"Average {factor} without heart disease: {df[df['HeartDisease'] == 0][factor].mean():.1f}")

# Gender statistics
print("\nGender Statistics:")
print(f"Percentage of males with heart disease: {(df[(df['Gender'] == 1) & (df['HeartDisease'] == 1)].shape[0] / df[df['Gender'] == 1].shape[0] * 100):.1f}%")
print(f"Percentage of females with heart disease: {(df[(df['Gender'] == 0) & (df['HeartDisease'] == 1)].shape[0] / df[df['Gender'] == 0].shape[0] * 100):.1f}%")

# Age group statistics
print("\nAge Group Statistics:")
for age_group in df['AgeGroup'].unique():
    total = df[df['AgeGroup'] == age_group].shape[0]
    with_disease = df[(df['AgeGroup'] == age_group) & (df['HeartDisease'] == 1)].shape[0]
    print(f"{age_group}: {with_disease/total*100:.1f}% have heart disease")

print("\nAnalysis complete! Check the 'Visualizations/insights' directory for visualizations.") 