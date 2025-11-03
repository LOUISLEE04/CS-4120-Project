import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import mlflow
import mlflow.sklearn
from pathlib import Path

mlruns_path = str(Path().resolve().parent / "mlruns")
mlflow.set_tracking_uri(mlruns_path)
mlflow.set_experiment("DecisionTree_Baseline")

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
with mlflow.start_run(run_name="DecisionTree_depth10"):
    
    mlflow.log_param("model_type", "DecisionTreeRegressor")
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("min_samples_split", 2)
    mlflow.log_param("test_size", 0.3)
    mlflow.log_param("random_state", 42)
    
    dt_reg = DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=2,
        random_state=42
    )
    dt_reg.fit(X_train, y_train)
    
    y_pred = dt_reg.predict(X_test)
    
    mae = metrics.mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_rmse", rmse)
    mlflow.sklearn.log_model(dt_reg, "model")
    
    print(f"Decision Tree Regressor Results:")
    print(f"Test MAE:  ${mae:.2f}")
    print(f"Test RMSE: ${rmse:.2f}")

    print(f"\n✓ Results saved to MLflow at: {mlruns_path}")
