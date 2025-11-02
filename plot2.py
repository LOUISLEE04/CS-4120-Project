import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('insurance.csv')

feature_cols = ['age', 'bmi', 'children', 'charges']

numeric_data = data[feature_cols]

correlation_matrix = numeric_data.corr()
plt.figure(figsize=(10, 8))

sns.heatmap(correlation_matrix, 
            annot=True,      
            fmt='.3f',       
            cmap='coolwarm', 
            square=True)     

plt.title('Correlation Heatmap - Insurance Dataset')
plt.tight_layout()
plt.savefig('plot2_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

