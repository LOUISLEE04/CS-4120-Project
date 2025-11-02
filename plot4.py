import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv('insurance.csv')

data_encoded = data.copy()
data_encoded['sex'] = (data['sex'] == 'male').astype(int)
data_encoded['smoker'] = (data['smoker'] == 'yes').astype(int)

data_encoded['region_northwest'] = (data['region'] == 'northwest').astype(int)
data_encoded['region_southeast'] = (data['region'] == 'southeast').astype(int)
data_encoded['region_southwest'] = (data['region'] == 'southwest').astype(int)

feature_cols = ['age', 'sex', 'bmi', 'children', 'smoker',
                'region_northwest', 'region_southeast', 'region_southwest']
X = data_encoded[feature_cols]
y = data_encoded['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)

residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6, edgecolor='k', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Charges ($)')
plt.ylabel('Residuals ($)')
plt.title('Residuals vs Predicted Values - Linear Regression')
plt.tight_layout()
plt.savefig('plot4_residuals.png', dpi=300)
plt.show()
