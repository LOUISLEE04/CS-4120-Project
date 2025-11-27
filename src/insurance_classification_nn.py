import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
df = pd.read_csv('insurance.csv')

threshold = 20000
df['highCharges'] = (df['charges'] > threshold).astype(int)

X = df.drop(['charges', 'highCharges'], axis=1)
y = df['highCharges']

X_encoded = pd.get_dummies(X, columns=['sex', 'smoker', 'region'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, 
    test_size=0.2, 
    random_state=RANDOM_STATE,
    stratify=y  
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_features = X_train_scaled.shape[1]

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(n_features,), 
                 name='hidden_layer_1'),
    
    layers.Dropout(0.3, name='dropout_1'),
    
    layers.Dense(32, activation='relu', name='hidden_layer_2'),
    
    layers.Dropout(0.3, name='dropout_2'),
    
    layers.Dense(1, activation='sigmoid', name='output_layer')
])

model.summary()

model.compile(
    optimizer='adam',  
    loss='binary_crossentropy',  
    metrics=['accuracy']  
)

history = model.fit(
    X_train_scaled, 
    y_train,
    epochs=100,  
    batch_size=32,  
    validation_split=0.2,  
    verbose=1  
)

y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

baseline_roc_auc = 0.900
baseline_f1 = 0.800

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Best Classification Model\n(Neural Network)', 
          fontsize=14, fontweight='bold')
plt.colorbar()

labels = ['Low Charges', 'High Charges']
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black',
                 fontsize=14)

plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('plot3_confusion_matrix.png', dpi=300, bbox_inches='tight')

plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('learning_curves_classification.png', dpi=300, bbox_inches='tight')
print("✓ Learning curves saved as 'learning_curves_classification.png'")
plt.show()

model.save('insurance_classification_model.h5')

nn_is_better = (f1 > baseline_f1) or (roc_auc > baseline_roc_auc)
if nn_is_better:
    improvement_f1 = ((f1 - baseline_f1) / baseline_f1) * 100
    improvement_auc = ((roc_auc - baseline_roc_auc) / baseline_roc_auc) * 100
else:
    print(f"\n✗ Baseline model (Naive Bayes) performs better")
    print(f"   Consider: hyperparameter tuning, more epochs, or different architecture")
