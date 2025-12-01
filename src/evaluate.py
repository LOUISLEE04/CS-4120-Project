# evaluate.py

import mlflow
import joblib
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_filepath
import data
from data import load_clean_split
import pandas as pd
from sklearn.inspection import permutation_importance
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from train_nn import CreateMlpRegressor

def evaluate_best_classification_baseline():
    """
    Evaluate best model and log to MLflow
    """
    # Load test data
    X_train, X_test, y_train, y_test = load_clean_split(isClassification=True)

    # Hardcode your best model after looking at training results
    best_model_path = "models/naive_bayes/gnb_var1e-07_balanced.pkl"
    best_run_id = "abc123def456"  # Copy this from MLflow UI or training output

    model = joblib.load(best_model_path)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    # Metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== Best Model Test Results ===")
    print(f"Model: {best_model_path}")
    print(f"Accuracy: {test_accuracy:.3f}")
    print(f"F1 Score: {test_f1:.3f}")
    print(f"ROC-AUC: {test_roc_auc:.3f}")

    # ===== Log to MLflow (reopens the same run from training) =====
    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_roc_auc", test_roc_auc)

        # Save confusion matrix plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix - Test Set')
        plt.tight_layout()
        plt.savefig('plots/confusion_matrix_nb.png', dpi=300)
        mlflow.log_artifact('plots/confusion_matrix_nb.png')
        plt.close()

        print(f"\n✓ Test metrics logged to MLflow run: {best_run_id}")

    return test_accuracy, test_f1, test_roc_auc



def evaluate_mlp_regressor():
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)

    X_train, X_test, y_train, y_test = data.load_clean_split(isClassification=False)

    # Load the saved best model
    model_path = get_filepath('models', 'best_MlpRegressor')
    best_model = joblib.load(model_path)

    # Make predictions on test set
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print("=" * 60)

    # Calculate residuals
    residuals = y_test - y_pred

    # Create residual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual')
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.title('Residual Plot: MLPRegressor', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_path = get_filepath('models', 'mlp_residual_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nResidual plot saved to: {plot_path}")
    plt.show()

    # Optional: Additional diagnostic info
    print("\nRESIDUAL STATISTICS")
    print("=" * 60)
    print(f"Mean of residuals:     {np.mean(residuals):.4f}")
    print(f"Std dev of residuals:  {np.std(residuals):.4f}")
    print(f"Min residual:          {np.min(residuals):.4f}")
    print(f"Max residual:          {np.max(residuals):.4f}")

    return best_model, y_pred, residuals


def analyze_feature_importance():
    from sklearn.inspection import permutation_importance

    X_train, X_test, y_train, y_test = data.load_clean_split(isClassification=False)
    model_path = get_filepath('models', 'best_MlpRegressor')
    best_model = joblib.load(model_path)

    # Permutation importance
    perm_importance = permutation_importance(
        best_model, X_test, y_test,
        n_repeats=10, random_state=42, n_jobs=-1
    )

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)

    print("\nFEATURE IMPORTANCE")
    print("=" * 60)
    print(importance_df)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance_mean'])
    plt.xlabel('Importance')
    plt.title('Feature Importance (Permutation)')
    plt.tight_layout()
    plt.savefig(get_filepath('plots', 'feature_importance.png'), dpi=300)
    plt.show()


def create_predictions_csv():
    # Load your test data
    X_train, X_test, y_train, y_test = data.load_clean_split(isClassification=False)

    # Load the model
    model_path = get_filepath('models', 'best_MlpRegressor')
    best_model = joblib.load(model_path)

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Calculate residuals
    residuals = y_test - y_pred

    # Create DataFrame with all info
    results_df = X_test.copy()
    results_df['actual_charges'] = y_test
    results_df['predicted_charges'] = y_pred
    results_df['residual'] = residuals
    results_df['absolute_error'] = np.abs(residuals)
    results_df['percent_error'] = (residuals / y_test) * 100

    # Sort by residual to see biggest errors first
    results_df = results_df.sort_values('residual', ascending=False)

    # Save to CSV
    output_path = get_filepath('results', 'mlp_predictions.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

    # Show some interesting statistics
    print("\n" + "=" * 60)
    print("TOP 10 UNDERESTIMATIONS (Model predicted too low)")
    print("=" * 60)
    print(results_df.head(10)[['age', 'bmi', 'children', 'smoker',
                               'actual_charges', 'predicted_charges', 'residual']])

    print("\n" + "=" * 60)
    print("TOP 10 OVERESTIMATIONS (Model predicted too high)")
    print("=" * 60)
    print(results_df.tail(10)[['age', 'bmi', 'children', 'smoker',
                               'actual_charges', 'predicted_charges', 'residual']])

    return results_df


def FeatureImportance():

    # Load model and data
    model = joblib.load(get_filepath("models",'best_MLP_regressor'))
    X_train, X_test, y_train, y_test = load_clean_split(isClassification=False)


    feature_names = [
        'age', 'bmi', 'children',
        'smoker_yes',
        'sex_male',
        'region_northwest', 'region_southeast', 'region_southwest'
    ]

    # Calculate importance
    r = permutation_importance(
        model, X_test, y_test,
        n_repeats=30,
        random_state=42
    )

    # Print sorted results
    print("Feature Importance:")
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{feature_names[i]:<20} {r.importances_mean[i]:.4f} +/- {r.importances_std[i]:.4f}")

if __name__ == "__main__":
    #model, predictions, residuals = evaluate_mlp_regressor()
    #analyze_feature_importance()
    #create_predictions_csv()
    #evaluate_best_classification_baseline()
    FeatureImportance()