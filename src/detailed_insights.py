import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from matplotlib.patches import Patch

# Create output directory if it doesn't exist
os.makedirs('../Visualizations/detailed_insights', exist_ok=True)

# Load the data
data_path = r'C:\Users\Addison Thompson\Downloads\Heart_Predictions_Dataset\Heart Prediction Quantum Dataset.csv'
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip('`')

# Set style for better visualizations
plt.style.use('ggplot')
sns.set_style("whitegrid")

# 1. Detailed Age Analysis
plt.figure(figsize=(15, 10))

# Age distribution with density plot
plt.subplot(2, 2, 1)
sns.kdeplot(data=df, x='Age', hue='HeartDisease', fill=True)
plt.title('Age Distribution Density Plot')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend(['No Heart Disease', 'Heart Disease'])

# Age vs Heart Disease Probability
plt.subplot(2, 2, 2)
# Modified to only include age groups up to 80
age_groups = pd.cut(df['Age'], bins=range(30, 81, 5))
age_prob = df.groupby(age_groups)['HeartDisease'].mean()
# Remove any empty age groups
age_prob = age_prob[age_prob.notna()]
ax = age_prob.plot(kind='bar')
plt.title('Probability of Heart Disease by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Probability of Heart Disease')

# Create custom labels for age groups
labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in age_prob.index]
ax.set_xticklabels(labels, rotation=45)

# Ensure we only show actual data points
plt.xlim(-0.5, len(age_prob)-0.5)

# Age vs Risk Factors
plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='Age', y='BloodPressure', hue='HeartDisease', alpha=0.6)
plt.title('Age vs Blood Pressure')
plt.xlabel('Age')
plt.ylabel('Blood Pressure')

plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='Age', y='Cholesterol', hue='HeartDisease', alpha=0.6)
plt.title('Age vs Cholesterol')
plt.xlabel('Age')
plt.ylabel('Cholesterol')

plt.tight_layout()
plt.savefig('../Visualizations/detailed_insights/age_analysis.png')
plt.close()

# 2. Risk Factor Interactions
plt.figure(figsize=(15, 10))

# Blood Pressure vs Cholesterol with Age as size
plt.subplot(2, 2, 1)
colors = {0: 'steelblue', 1: 'salmon'}
bp_chol_plot = sns.scatterplot(data=df, x='BloodPressure', y='Cholesterol', 
                hue='HeartDisease', size='Age', alpha=0.6, palette=colors)
plt.title('Blood Pressure vs Cholesterol (Size: Age)')
plt.xlabel('Blood Pressure')
plt.ylabel('Cholesterol')
# Create custom legend handles
legend_elements = [Patch(facecolor='steelblue', label='No'),
                  Patch(facecolor='salmon', label='Yes')]
bp_chol_plot.legend(handles=legend_elements, title='Heart Disease', loc='upper left')

# Heart Rate vs Blood Pressure
plt.subplot(2, 2, 2)
hr_bp_plot = sns.scatterplot(data=df, x='HeartRate', y='BloodPressure', 
                hue='HeartDisease', alpha=0.6, palette=colors)
plt.title('Heart Rate vs Blood Pressure')
plt.xlabel('Heart Rate')
plt.ylabel('Blood Pressure')
hr_bp_plot.legend(handles=legend_elements, title='Heart Disease', loc='upper left')

# Risk Factor Distributions by Age Group
plt.subplot(2, 2, 3)
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 40, 50, 60, 70, 100], 
                       labels=['<40', '40-50', '50-60', '60-70', '70+'])
bp_plot = sns.boxplot(data=df, x='AgeGroup', y='BloodPressure', hue='HeartDisease', palette=colors)
plt.title('Blood Pressure Distribution by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Blood Pressure')
bp_plot.legend(handles=legend_elements, title='Heart Disease', loc='upper left')

plt.subplot(2, 2, 4)
chol_plot = sns.boxplot(data=df, x='AgeGroup', y='Cholesterol', hue='HeartDisease', palette=colors)
plt.title('Cholesterol Distribution by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Cholesterol')
chol_plot.legend(handles=legend_elements, title='Heart Disease', loc='upper left')

plt.tight_layout()
plt.savefig('../Visualizations/detailed_insights/risk_factor_interactions.png')
plt.close()

# 3. Statistical Tests
print("\n=== Detailed Statistical Analysis ===")

# Age differences
age_with_disease = df[df['HeartDisease'] == 1]['Age']
age_without_disease = df[df['HeartDisease'] == 0]['Age']
t_stat, p_value = stats.ttest_ind(age_with_disease, age_without_disease)
print(f"\nAge Difference Test:")
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.3f}")
print("The difference in age between groups is statistically significant" if p_value < 0.05 else "The difference in age between groups is not statistically significant")

# Risk factor differences
print("\nRisk Factor Differences:")
for factor in ['BloodPressure', 'Cholesterol', 'HeartRate']:
    with_disease = df[df['HeartDisease'] == 1][factor]
    without_disease = df[df['HeartDisease'] == 0][factor]
    t_stat, p_value = stats.ttest_ind(with_disease, without_disease)
    print(f"\n{factor} Difference Test:")
    print(f"T-statistic: {t_stat:.3f}")
    print(f"P-value: {p_value:.3f}")
    print(f"The difference in {factor} between groups is statistically significant" if p_value < 0.05 else f"The difference in {factor} between groups is not statistically significant")

# 4. Risk Factor Correlations by Age Group
print("\nRisk Factor Correlations by Age Group:")
for age_group in df['AgeGroup'].unique():
    group_data = df[df['AgeGroup'] == age_group]
    correlation = group_data[['BloodPressure', 'Cholesterol', 'HeartRate', 'HeartDisease']].corr()
    print(f"\n{age_group} Age Group Correlations:")
    print(correlation['HeartDisease'].sort_values(ascending=False))

# 5. Gender and Age Interaction
plt.figure(figsize=(12, 8))
# Create custom color mapping
colors = {0: 'salmon', 1: 'steelblue'}
# Create the boxplot with specific colors
ax = sns.boxplot(data=df, x='AgeGroup', y='Age', hue='Gender', palette=colors)
plt.title('Age Distribution by Gender and Age Group')
plt.xlabel('Age Group')
plt.ylabel('Age')

# Ensure x-axis labels are visible
plt.xticks(rotation=0)

# Create custom legend handles
legend_elements = [Patch(facecolor='salmon', label='Female'),
                  Patch(facecolor='steelblue', label='Male')]
# Update legend with custom handles and position
ax.legend(handles=legend_elements, title='Gender', loc='upper left')

# Adjust y-axis limits
plt.ylim(25, 85)

# Ensure the x-axis labels are not cut off
plt.tight_layout(pad=2)
plt.savefig('../Visualizations/detailed_insights/gender_distribution.png', bbox_inches='tight', dpi=300)
plt.close()

# Print the exact numbers for verification
print("\nGender Distribution by Age Group:")
for age_group in df['AgeGroup'].unique():
    group_data = df[df['AgeGroup'] == age_group]
    total = len(group_data)
    females = len(group_data[group_data['Gender'] == 0])
    males = len(group_data[group_data['Gender'] == 1])
    print(f"\n{age_group}:")
    print(f"Total: {total}")
    print(f"Females: {females} ({females/total*100:.1f}%)")
    print(f"Males: {males} ({males/total*100:.1f}%)")

# 6. Risk Factor Thresholds
print("\nRisk Factor Thresholds:")
for factor in ['BloodPressure', 'Cholesterol', 'HeartRate']:
    # Calculate optimal threshold using ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(df['HeartDisease'], df[factor])
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"\n{factor} Optimal Threshold: {optimal_threshold:.1f}")
    
    # Calculate sensitivity and specificity
    predictions = (df[factor] > optimal_threshold).astype(int)
    sensitivity = np.mean(predictions[df['HeartDisease'] == 1] == 1)
    specificity = np.mean(predictions[df['HeartDisease'] == 0] == 0)
    print(f"Sensitivity: {sensitivity:.2%}")
    print(f"Specificity: {specificity:.2%}")

print("\nDetailed analysis complete! Check the 'Visualizations/detailed_insights' directory for visualizations.") 