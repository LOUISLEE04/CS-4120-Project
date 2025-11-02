from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

from data import load_data, clean_data, split_data
import features

# ----- CLASSIFICATION BASELINES -----
def load_and_split_classification_data():
    classificationData = clean_data(load_data())
    classificationData = features.add_highCharges(classificationData)
    classificationData = features.encode(classificationData)
    return split_data(classificationData, isClassification=True)

# --- Naive Bayes Baseline ---
from sklearn.naive_bayes import GaussianNB

def run_naive_bayes(var_smoothing, upsample: bool):
    X_train, X_test, y_train, y_test = load_and_split_classification_data()
    mlflow.set_experiment("NaiveBayes_Baseline")

    if upsample:
        balance="balanced"
        X_train, y_train = features.upsample(X_train, y_train)

    with mlflow.start_run(run_name=f"GaussianNB_var{var_smoothing}_imbalanced"):

        gnb = GaussianNB(var_smoothing=1e-12)
        mlflow.log_param("model_type", "GaussianNB")
        # - Fit and Predict -
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_test)
        y_proba = gnb.predict_proba(X_test)[:,1]

        # - Metrics -
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(gnb, "model")

        print(f"F1 Score: {f1:.3f}")
        print(f"ROC AUC: {roc_auc:.6f}")

