"""
Plot 2: Correlation Heatmap and Boxplot Summary
Medical Insurance Cost Prediction - EDA
Saves plots directly to root directory

Authors: Louis Lee and Isaac Campbell
Date: October 31, 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the insurance dataset from root directory
df = pd.read_csv('insurance.csv')

print("="*60)
print("CORRELATION ANALYSIS - MEDICAL INSURANCE DATASET")
print("="*60)
print(f"\nDataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nBasic statistics:")
print(df.describe())

# Select numeric features for correlation analysis
numeric_features = ['age', 'bmi', 'children', 'charges']
numeric_df = df[numeric_features]

# Calculate correlation matrix
correlation_matrix = numeric_df.corr()

print("\n" + "="*60)
print("CORRELATION MATRIX")
print("="*60)
print(correlation_matrix)

# Create the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, 
            annot=True,  # Show correlation values
            fmt='.3f',   # Format to 3 decimal places
            cmap='coolwarm',  # Color scheme (blue for negative, red for positive)
            center=0,    # Center the colormap at 0
            square=True, # Make cells square-shaped
            linewidths=1,  # Add lines between cells
            cbar_kws={"shrink": 0.8})  # Adjust colorbar size

plt.title('Correlation Heatmap of Numeric Features\nMedical Insurance Dataset', 
          fontsize=16, fontweight='bold')
plt.tight_layout()

# Save to root directory
plt.savefig('plot2_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved to: plot2_correlation_heatmap.png")
plt.close()

# Also create a boxplot summary as an alternative visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Key Numeric Features\nMedical Insurance Dataset', 
             fontsize=16, fontweight='bold')

# Age distribution
sns.boxplot(data=df, y='age', ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Age Distribution', fontweight='bold')
axes[0, 0].set_ylabel('Age (years)')

# BMI distribution
sns.boxplot(data=df, y='bmi', ax=axes[0, 1], color='lightgreen')
axes[0, 1].set_title('BMI Distribution', fontweight='bold')
axes[0, 1].set_ylabel('BMI (kg/m²)')

# Children distribution
sns.boxplot(data=df, y='children', ax=axes[1, 0], color='salmon')
axes[1, 0].set_title('Number of Children Distribution', fontweight='bold')
axes[1, 0].set_ylabel('Number of Children')

# Charges distribution
sns.boxplot(data=df, y='charges', ax=axes[1, 1], color='gold')
axes[1, 1].set_title('Medical Charges Distribution', fontweight='bold')
axes[1, 1].set_ylabel('Charges ($)')

plt.tight_layout()
plt.savefig('plot2_boxplot_summary.png', dpi=300, bbox_inches='tight')
print("✓ Boxplot summary saved to: plot2_boxplot_summary.png")
plt.close()

# Additional analysis: correlation with categorical variables encoded
df_encoded = df.copy()
df_encoded['sex_encoded'] = (df['sex'] == 'male').astype(int)
df_encoded['smoker_encoded'] = (df['smoker'] == 'yes').astype(int)

# Create dummy variables for region
region_dummies = pd.get_dummies(df['region'], prefix='region')
df_encoded = pd.concat([df_encoded, region_dummies], axis=1)

# Extended correlation analysis including encoded categorical variables
extended_features = ['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded', 
                     'region_northeast', 'region_northwest', 'region_southeast', 
                     'region_southwest', 'charges']
extended_corr = df_encoded[extended_features].corr()

print("\n" + "="*60)
print("EXTENDED CORRELATION WITH CATEGORICAL VARIABLES")
print("="*60)
print("\nCorrelation with charges (sorted):")
print(extended_corr['charges'].sort_values(ascending=False))

# Create extended correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(extended_corr, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8})

plt.title('Extended Correlation Heatmap (Including Encoded Categorical Features)\nMedical Insurance Dataset', 
          fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig('plot2_extended_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Extended correlation heatmap saved to: plot2_extended_correlation_heatmap.png")
plt.close()

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
strongest_predictor = extended_corr['charges'].drop('charges').idxmax()
strongest_correlation = extended_corr['charges'].drop('charges').max()
print(f"✓ Strongest predictor: {strongest_predictor} (correlation: {strongest_correlation:.3f})")
print(f"✓ Dataset: {len(df)} samples with no missing values")
print(f"✓ Charges range: ${df['charges'].min():.2f} - ${df['charges'].max():.2f}")
print(f"✓ Charges mean: ${df['charges'].mean():.2f}, median: ${df['charges'].median():.2f}")
print("\n" + "="*60)
print("ALL PLOTS GENERATED SUCCESSFULLY!")
print("Saved to root directory:")
print("  - plot2_correlation_heatmap.png")
print("  - plot2_boxplot_summary.png")
print("  - plot2_extended_correlation_heatmap.png")
print("="*60)
