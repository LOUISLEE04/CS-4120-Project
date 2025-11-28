# evaluate.py

import mlflow
import joblib
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from data import load_clean_split


def evaluate_best_model():
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

if __name__ == "__main__":
    evaluate_best_model()
