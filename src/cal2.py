import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv('insurance.csv')

threshold = 20000
df['highCharges'] = (df['charges'] > threshold).astype(int)

X = df.drop(['charges', 'highCharges'], axis=1)
y = df['highCharges']

X_encoded = pd.get_dummies(X, columns=['sex', 'smoker', 'region'], drop_first=True)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
)

nb = GaussianNB(var_smoothing=1e-9)
nb.fit(X_train, y_train)

y_val_pred = nb.predict(X_val)
y_val_proba = nb.predict_proba(X_val)[:, 1]
y_test_pred = nb.predict(X_test)
y_test_proba = nb.predict_proba(X_test)[:, 1]

nb_val_acc = accuracy_score(y_val, y_val_pred)
nb_val_f1 = f1_score(y_val, y_val_pred)
nb_val_roc = roc_auc_score(y_val, y_val_proba)
nb_test_acc = accuracy_score(y_test, y_test_pred)
nb_test_f1 = f1_score(y_test, y_test_pred)
nb_test_roc = roc_auc_score(y_test, y_test_proba)

dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)

y_val_pred = dt.predict(X_val)
y_val_proba = dt.predict_proba(X_val)[:, 1]
y_test_pred = dt.predict(X_test)
y_test_proba = dt.predict_proba(X_test)[:, 1]

dt_val_acc = accuracy_score(y_val, y_val_pred)
dt_val_f1 = f1_score(y_val, y_val_pred)
dt_val_roc = roc_auc_score(y_val, y_val_proba)
dt_test_acc = accuracy_score(y_test, y_test_pred)
dt_test_f1 = f1_score(y_test, y_test_pred)
dt_test_roc = roc_auc_score(y_test, y_test_proba)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, 
          validation_data=(X_val_scaled, y_val), verbose=0)

y_val_proba = model.predict(X_val_scaled, verbose=0)
y_val_pred = (y_val_proba > 0.5).astype(int).flatten()
y_test_proba = model.predict(X_test_scaled, verbose=0)
y_test_pred = (y_test_proba > 0.5).astype(int).flatten()

nn_val_acc = accuracy_score(y_val, y_val_pred)
nn_val_f1 = f1_score(y_val, y_val_pred)
nn_val_roc = roc_auc_score(y_val, y_val_proba)
nn_test_acc = accuracy_score(y_test, y_test_pred)
nn_test_f1 = f1_score(y_test, y_test_pred)
nn_test_roc = roc_auc_score(y_test, y_test_proba)

print("\nNaive Bayes:")
print(f"  Validation - Acc: {nb_val_acc:.3f}, F1: {nb_val_f1:.3f}, ROC-AUC: {nb_val_roc:.3f}")
print(f"  Test - Acc: {nb_test_acc:.3f}, F1: {nb_test_f1:.3f}, ROC-AUC: {nb_test_roc:.3f}")

print("\nDecision Tree:")
print(f"  Validation - Acc: {dt_val_acc:.3f}, F1: {dt_val_f1:.3f}, ROC-AUC: {dt_val_roc:.3f}")
print(f"  Test - Acc: {dt_test_acc:.3f}, F1: {dt_test_f1:.3f}, ROC-AUC: {dt_test_roc:.3f}")

print("\nNeural Network:")
print(f"  Validation - Acc: {nn_val_acc:.3f}, F1: {nn_val_f1:.3f}, ROC-AUC: {nn_val_roc:.3f}")
print(f"  Test - Acc: {nn_test_acc:.3f}, F1: {nn_test_f1:.3f}, ROC-AUC: {nn_test_roc:.3f}")
