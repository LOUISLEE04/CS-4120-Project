"""
Regression Baseline Models for Medical Insurance Cost Prediction
CS-4120 Project - Louis Lee and Isaac Campbell

This script:
1. Loads and preprocesses the insurance dataset
2. Splits data into train/validation/test sets
3. Trains baseline regression models (Linear Regression & Decision Tree)
4. Tracks experiments with MLflow
5. Generates Plot 4: Residuals vs Predicted for best model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the insurance dataset.
    
    Args:
        filepath: Path to the insurance.csv file
        
    Returns:
        X: Feature matrix
        y: Target variable (charges)
        feature_names: List of feature names
    """
    print("="*60)
    print("STEP 1: Loading and Preprocessing Data")
    print("="*60)
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values
    missing = df.isnull().sum()
    print(f"\nMissing values:\n{missing}")
    
    # Basic statistics
    print(f"\nTarget variable (charges) statistics:")
    print(df['charges'].describe())
    
    # Encode categorical variables
    df_encoded = df.copy()
    
    # Binary encoding for sex and smoker
    df_encoded['sex'] = LabelEncoder().fit_transform(df_encoded['sex'])
    df_encoded['smoker'] = LabelEncoder().fit_transform(df_encoded['smoker'])
    
    # One-hot encoding for region
    df_encoded = pd.get_dummies(df_encoded, columns=['region'], drop_first=True)
    
    print(f"\nEncoded dataset shape: {df_encoded.shape}")
    print(f"Features after encoding: {df_encoded.columns.tolist()}")
    
    # Separate features and target
    X = df_encoded.drop('charges', axis=1)
    y = df_encoded['charges']
    
    feature_names = X.columns.tolist()
    
    return X, y, feature_names

def split_data(X, y, random_seed=RANDOM_SEED):
    """
    Split data into train, validation, and test sets.
    
    Train: 70%
    Validation: 15%
    Test: 15%
    
    Args:
        X: Feature matrix
        y: Target variable
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("\n" + "="*60)
    print("STEP 2: Splitting Data")
    print("="*60)
    
    # First split: 70% train, 30% temp (which will be split into val and test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_seed
    )
    
    # Second split: Split temp into 50-50 (15% val, 15% test of original)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_seed
    )
    
    print(f"\nTrain set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set size: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_and_evaluate_model(model, model_name, X_train, X_val, X_test, 
                             y_train, y_val, y_test, feature_names):
    """
    Train and evaluate a regression model.
    
    Args:
        model: Sklearn model instance
        model_name: Name of the model for logging
        X_train, X_val, X_test: Feature sets
        y_train, y_val, y_test: Target sets
        feature_names: List of feature names
        
    Returns:
        Tuple of (model, metrics_dict)
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=model_name):
        
        # Train model
        model.fit(X_train, y_train)
        print(f"✓ Model trained successfully")
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics for all sets
        metrics = {}
        
        for split, y_true, y_pred in [
            ('train', y_train, y_train_pred),
            ('val', y_val, y_val_pred),
            ('test', y_test, y_test_pred)
        ]:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            metrics[f'{split}_MAE'] = mae
            metrics[f'{split}_RMSE'] = rmse
            metrics[f'{split}_R2'] = r2
            
            # Log to MLflow
            mlflow.log_metric(f"{split}_MAE", mae)
            mlflow.log_metric(f"{split}_RMSE", rmse)
            mlflow.log_metric(f"{split}_R2", r2)
        
        # Log model parameters
        if hasattr(model, 'get_params'):
            mlflow.log_params(model.get_params())
        
        # Log the model
        mlflow.sklearn.log_model(model, model_name)
        
        # Print results
        print(f"\n{'Validation Set Metrics':^30}")
        print(f"MAE:  ${metrics['val_MAE']:,.2f}")
        print(f"RMSE: ${metrics['val_RMSE']:,.2f}")
        print(f"R²:   {metrics['val_R2']:.4f}")
        
        print(f"\n{'Test Set Metrics':^30}")
        print(f"MAE:  ${metrics['test_MAE']:,.2f}")
        print(f"RMSE: ${metrics['test_RMSE']:,.2f}")
        print(f"R²:   {metrics['test_R2']:.4f}")
    
    return model, metrics

def create_residual_plots(model, model_name, X_test, y_test, save_path='plot4_residuals.png'):
    """
    Create Plot 4: Residuals vs Predicted (with histogram)
    
    Args:
        model: Trained model
        model_name: Name of the model
        X_test: Test features
        y_test: Test targets
        save_path: Path to save the plot
    """
    print(f"\n{'='*60}")
    print(f"Creating Plot 4: Residual Plots for {model_name}")
    print(f"{'='*60}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate residuals
    residuals = y_test - y_pred
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero residual line')
    axes[0].set_xlabel('Predicted Charges ($)', fontsize=12)
    axes[0].set_ylabel('Residuals ($)', fontsize=12)
    axes[0].set_title(f'Residuals vs Predicted Values\n{model_name}', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add horizontal lines at ±1 std
    std_residual = np.std(residuals)
    axes[0].axhline(y=std_residual, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'±1 STD (${std_residual:,.0f})')
    axes[0].axhline(y=-std_residual, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Plot 2: Histogram of Residuals
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero residual')
    axes[1].set_xlabel('Residuals ($)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Distribution of Residuals\n{model_name}', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add statistics text box
    stats_text = f'Mean: ${np.mean(residuals):,.2f}\nStd: ${np.std(residuals):,.2f}\nMedian: ${np.median(residuals):,.2f}'
    axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {save_path}")
    
    # Calculate and print residual statistics
    print(f"\nResidual Statistics:")
    print(f"  Mean: ${np.mean(residuals):,.2f}")
    print(f"  Std:  ${np.std(residuals):,.2f}")
    print(f"  Min:  ${np.min(residuals):,.2f}")
    print(f"  Max:  ${np.max(residuals):,.2f}")
    
    return fig

def create_results_table(results):
    """
    Create a formatted results table.
    
    Args:
        results: Dictionary with model results
    """
    print("\n" + "="*60)
    print("TABLE 2: Regression Metrics for All Baselines")
    print("="*60)
    
    table_data = []
    for model_name, metrics in results.items():
        table_data.append({
            'Model': model_name,
            'Val MAE': f"${metrics['val_MAE']:,.2f}",
            'Val RMSE': f"${metrics['val_RMSE']:,.2f}",
            'Test MAE': f"${metrics['test_MAE']:,.2f}",
            'Test RMSE': f"${metrics['test_RMSE']:,.2f}",
        })
    
    df_results = pd.DataFrame(table_data)
    print("\n" + df_results.to_string(index=False))
    
    # Save to CSV
    df_results.to_csv('table2_regression_results.csv', index=False)
    print(f"\n✓ Results saved to: table2_regression_results.csv")
    
    return df_results

def main():
    """
    Main execution function.
    """
    print("\n" + "="*60)
    print("MEDICAL INSURANCE COST PREDICTION")
    print("Regression Baseline Models")
    print("="*60)
    
    # Set MLflow tracking
    mlflow.set_experiment("Medical_Insurance_Regression_Baselines")
    
    # Step 1: Load and preprocess data
    X, y, feature_names = load_and_preprocess_data('insurance.csv')
    
    # Step 2: Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Step 3: Train baseline models
    results = {}
    
    # Model 1: Linear Regression
    lr_model = LinearRegression()
    lr_model, lr_metrics = train_and_evaluate_model(
        lr_model, "Linear_Regression", 
        X_train, X_val, X_test, 
        y_train, y_val, y_test, 
        feature_names
    )
    results["Linear Regression"] = lr_metrics
    
    # Model 2: Decision Tree Regressor
    dt_model = DecisionTreeRegressor(random_state=RANDOM_SEED, max_depth=10)
    dt_model, dt_metrics = train_and_evaluate_model(
        dt_model, "Decision_Tree", 
        X_train, X_val, X_test, 
        y_train, y_val, y_test, 
        feature_names
    )
    results["Decision Tree"] = dt_metrics
    
    # Step 4: Determine best model based on validation RMSE
    best_model_name = min(results, key=lambda x: results[x]['val_RMSE'])
    best_model = lr_model if best_model_name == "Linear Regression" else dt_model
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Validation RMSE: ${results[best_model_name]['val_RMSE']:,.2f}")
    print(f"{'='*60}")
    
    # Step 5: Create Plot 4 for best model
    create_residual_plots(best_model, best_model_name, X_test, y_test)
    
    # Step 6: Create results table
    create_results_table(results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. plot4_residuals.png - Residual plots")
    print("  2. table2_regression_results.csv - Results table")
    print("  3. MLflow tracking data in ./mlruns directory")
    print("\nTo view MLflow UI, run: mlflow ui")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
