import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import data
from src.data import load_clean_split
from utils import get_filepath
from sklearn.compose import TransformedTargetRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import keras
from keras import layers

# At the top of your file, after imports
warnings.filterwarnings('ignore', category=ConvergenceWarning)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def MlpRegressorGridSearch():
    X_train, X_test, y_train, y_test = data.load_clean_split(isClassification=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False), ['sex', 'smoker', 'region']),
            ('num', StandardScaler(), ['age', 'bmi', 'children'])
        ])

    mlp = Pipeline([
        ('preprocessor', preprocessor),
        ("model", MLPRegressor(
            activation="relu",
            solver="adam",
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            random_state=RANDOM_STATE,
        ))
    ])

    mlp_with_target_scaling = TransformedTargetRegressor(
        regressor=mlp,
        transformer=StandardScaler()
    )

    param_grid = {
        'regressor__model__hidden_layer_sizes': [(64,), (64, 32), (128, 64)],
        'regressor__model__learning_rate_init': [0.001, 0.01, 0.0001],
        'regressor__model__alpha': [0.01, 0.1, 0.2, 0.3, 0.4],
    }

    grid_search = GridSearchCV(
        mlp_with_target_scaling,
        param_grid,
        cv=5,
        scoring={
            'mse':'neg_mean_squared_error',
            'mae':'neg_mean_absolute_error',
        }, # Track mse & mae
        refit='mse', # Select model with best MSE
        verbose = 2,
        n_jobs = -1,  # use all cores
        random_state=RANDOM_STATE
    )

    print("Starting Grid Search...")
    grid_search.fit(X_train, y_train)
    best_mse = -grid_search.cv_results_['mean_test_mse'][grid_search.best_index_]
    best_mae = -grid_search.cv_results_['mean_test_mae'][grid_search.best_index_]

    print("\n" + "="*60)
    print("GRID SEARCH RESULTS")
    print("="*60)
    print("Best parameters:", grid_search.best_params_)
    print("Best CV score (MSE):", best_mse)
    print("Best CV score (RMSE):", np.sqrt(best_mse))
    print(f"Best CV score (MAE):  ${best_mae:,.2f}")
    print("=" * 60)

    # Get best model
    best_model = grid_search.best_estimator_

    # Extract the MLPRegressor from the wrapped pipeline
    mlp_model = best_model.regressor_.named_steps['model']

    # Get training and validation loss curves
    train_loss = mlp_model.loss_curve_
    val_loss = -np.array(mlp_model.validation_scores_)  # Negate because they're stored as negative

    print(f"\nTraining stopped after {len(train_loss)} epochs")
    print(f"Final training loss: {train_loss[-1]:.6f}")
    print(f"Final validation loss: {val_loss[-1]:.6f}")

    # Save the best model
    model_path = get_filepath('models', 'best_MlpRegressor')
    joblib.dump(best_model, model_path)
    print(f"\nBest model saved to: {model_path}")

    # Save grid search results
    results_path = get_filepath('models', 'MlpRegressor_GridSearchCV')
    joblib.dump(grid_search, results_path)
    print(f"Grid search results saved to: {results_path}")


    # ---- Plotting validation learning curve ----
    # After grid search, get best parameters
    best_params = grid_search.best_params_

    # Split training data into train/val
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.1, random_state=RANDOM_STATE
    )

    # Scale target - convert Series to numpy array first
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_split.values.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val_split.values.reshape(-1, 1)).ravel()

    # Retrain with best parameters to track both losses
    best_mlp = Pipeline([
        ('preprocessor', preprocessor),
        ("model", MLPRegressor(
            hidden_layer_sizes=best_params['regressor__model__hidden_layer_sizes'],
            learning_rate_init=best_params['regressor__model__learning_rate_init'],
            alpha=best_params['regressor__model__alpha'],
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=RANDOM_STATE,
            warm_start=False
        ))
    ])

    # Train and manually track validation loss
    train_losses = []
    val_losses = []

    for epoch in range(1000):
        # Set max_iter to current epoch + 1
        best_mlp.named_steps['model'].max_iter = epoch + 1
        best_mlp.fit(X_train_split, y_train_scaled)

        # Get training loss
        train_loss = best_mlp.named_steps['model'].loss_
        train_losses.append(train_loss)

        # Calculate validation loss
        y_val_pred = best_mlp.predict(X_val_split)
        val_loss = np.mean((y_val_scaled - y_val_pred) ** 2)
        val_losses.append(val_loss)

        # Early stopping check
        if epoch > 50 and val_losses[-1] > min(val_losses[-50:]):
            break

    # Plot both
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE on scaled target)', fontsize=12)
    plt.title('Learning Curves - Best Model', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mlp_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


def CreateMlpRegressor(layer1:int, layer2:int, dropout1:float, dropout2:float):
    """
    Finds the best MlP regressor
    """
    X_train, X_test, y_train, y_test = data.load_clean_split(isClassification=False)
    X_train = pd.get_dummies(X_train, columns=['sex', 'smoker', 'region'], drop_first=True) # Encode categorical variables


    model = keras.Sequential([
        layers.Dense(layer1, activation='relu', input_shape=(X_train.shape[1],),name='hidden_layer_1'),
        layers.Dropout(dropout1, name='dropout_1'),
        layers.Dense(layer2, activation='relu', name='hidden_layer_2'),
        layers.Dropout(dropout2, name='dropout_2'),
        layers.Dense(1, activation='linear', name='output_layer')
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=[keras.metrics.RootMeanSquaredError(),
                 keras.metrics.MeanSquaredError(),
                 keras.metrics.MeanAbsoluteError()]
    )

    return model


def MLPRegressor_withDropout_CV():
    keras_reg = KerasRegressor(
        model=CreateMlpRegressor,
        epochs=100,
        batch_size=32,
        verbose=0,
        validation_split=0.1,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,  # Stop if no improvement for 20 epochs
            restore_best_weights=True
        )],
        # Specify defaults for model parameters
        dropout1=0.2,
        dropout2=0.2,
        layer1=64,
        layer2=32
    )

    # Load data and encode
    X_train, X_test, y_train, y_test = load_clean_split(isClassification=False)

    # Define Pipeline
    numeric_features = ['age', 'bmi', 'children']
    categorical_features = ['smoker', 'sex', 'region']

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(drop='first', sparse_output=False), categorical_features),
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', keras_reg)
    ])

    param_grid = {
        'model__dropout1': [0.0, 0.2, 0.4],
        'model__dropout2': [0.0, 0.2, 0.4],
        'model__layer1': [64],
        'model__layer2': [32],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=1,
        verbose=2,
    )

    # Fit
    print("Starting grid search with cross-validation...")
    grid_search.fit(X_train, y_train)

    # Best model results
    print("\n" + "=" * 60)
    print("BEST MODEL - CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print("Best parameters:", grid_search.best_params_)
    print(f"Best CV MSE: {-grid_search.best_score_:.4f}")
    print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")

    # Get the best model
    best_model = grid_search.best_estimator_

    # Save the best model
    model_filename = get_filepath('models', 'best_MLP_regressor')
    joblib.dump(best_model, model_filename)
    print(f"\nBest model saved to: {model_filename}")



    return {
        'best_params': grid_search.best_params,
        'cv_mse': -grid_search.best_score_,
        'cv_rmse': np.sqrt(-grid_search.best_score_),
        'preprocessor': preprocessor,  # Need this for transforming data
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def plot_regression_learning_curves(cv_results):
    # Extract what we need
    best_params = cv_results['best_params']
    preprocessor = cv_results['preprocessor']
    X_train = cv_results['X_train']
    y_train = cv_results['y_train']

    # Extract hyperparameters
    best_dropout1 = best_params['model__dropout1']
    best_dropout2 = best_params['model__dropout2']
    best_layer1 = best_params['model__layer1']
    best_layer2 = best_params['model__layer2']

    print("\nRetraining best model to capture learning curves...")

    # Preprocess data
    X_train_processed = preprocessor.transform(X_train)

    # Create model with best hyperparameters
    final_model = CreateMlpRegressor(
        layer1=best_layer1,
        layer2=best_layer2,
        dropout1=best_dropout1,
        dropout2=best_dropout2
    )

    # Train and capture history
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    history = final_model.fit(
        X_train_processed, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,  # 20% for validation
        callbacks=[early_stopping],
        verbose=1
    )

    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Learning Curve - Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reg_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nLearning curves saved to 'reg_learning_curves.png'")


if __name__ == '__main__':
    MlpRegressorGridSearch()

