from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib

from data import load_clean_split
from utils import get_filepath
import features

# ----- CLASSIFICATION BASELINES -----
def load_and_split_encode_classification_data():
    classificationData = clean_data(load_data())
    classificationData = features.add_highCharges(classificationData)
    classificationData = features.encode(classificationData)
    y=classificationData['highCharges']
    X=classificationData.iloc[:, :-2]

    return train_test_split(X, y, test_size=0.3, random_state=42)


# --- Naive Bayes Baseline ---


def run_naive_bayes(var_smoothing, upsample: bool):
    X_train, X_test, y_train, y_test = load_clean_split(isClassification=True)
    mlflow.set_experiment("NaiveBayes_Baseline")

    if upsample:
        balance="balanced"
        X_train, y_train = features.upsample(X_train, y_train)
    else:
        balance="imbalanced"

    with mlflow.start_run(run_name=f"GaussianNB_var{var_smoothing}_{balance}") as run:

        gnb = GaussianNB(var_smoothing=var_smoothing)

        # Log params
        mlflow.log_param("model_type", "GaussianNB")
        mlflow.log_param("var_smoothing", var_smoothing)
        mlflow.log_param("upsampled", upsample)

        # ===== CV on training data =====
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1 = cross_val_score(gnb, X_train, y_train, cv=cv, scoring='f1')
        cv_roc_auc = cross_val_score(gnb, X_train, y_train, cv=cv, scoring='roc_auc')

        mlflow.log_metric("cv_f1_mean", cv_f1.mean())
        mlflow.log_metric("cv_f1_std", cv_f1.std())
        mlflow.log_metric("cv_roc_auc_mean", cv_roc_auc.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_roc_auc.std())

        gnb.fit(X_train, y_train)

        # Save the model and run id
        mlflow.sklearn.log_model(gnb, name="model")
        model_path = get_filepath("models", f"GNB_var{var_smoothing}_{balance}")
        joblib.dump(gnb, path)
        run_id = run.info.run_id

        print(f"--- GaussianNB_var{var_smoothing}_{balance} ---")
        print(f"CV ROC-AUC: {cv_roc_auc.mean():.3f} ± {cv_roc_auc.std():.3f}")

        return run_id, model_path, cv_roc_auc.mean()


def tune_naive_bayes(var_smoothing: list):
    results = []
    for var_smooth in var_smoothing:
        for upsample in [False, True]:
            run_id, path, cv_score = run_naive_bayes(var_smooth, upsample)
            results.append({'run_id': run_id, 'path': path, 'cv_score': cv_score})

    best = max(results, key=lambda x: x['cv_score'])
    print(f"Best model: {best['path']} with CV ROC-AUC: {best['cv_score']:.3f}")


tune_naive_bayes([10e-12, 10e-10, 10e-8, 10e-7])


# --- Decision Tree Classifier ---
from sklearn.tree import DecisionTreeClassifier

def run_decision_tree(max_depth, min_samples_split):
    X_train, X_test, y_train, y_test = load_and_split_classification_data()
    mlflow.set_experiment("DecisionTrees_Baseline")

    with mlflow.start_run(run_name=f"DecisionTree_maxDepth{max_depth}_MinSplit{min_samples_split}"):

        dtc = DecisionTreeClassifier(criterion='gini',max_depth=max_depth, min_samples_split=min_samples_split)
        mlflow.log_param("model_type", "DecisionTree")
        mlflow.log_param("criterion", 'gini')
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)

        # - Fit and Predict -
        dtc.fit(X_train, y_train)
        y_pred = dtc.predict(X_test)
        y_proba = dtc.predict_proba(X_test)[:,1]

        # - Metrics -
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("F1", f1)
        mlflow.sklearn.log_model(dtc, name="model")

        print(f"--- DecisionTree_maxDepth{max_depth}_MinSplit{min_samples_split} ---")
        print(f"F1 Score: {f1:.3f}")
        print(f"ROC AUC: {roc_auc:.6f}")

def tune_decision_tree(max_depth: list, min_split: list):
    for depth in max_depth:
        for value in min_split:
            run_decision_tree(depth, value)

# --- Run several models ---
tune_decision_tree([10, 50, 100], [2, 6])
def naive_bayes(var_smoothing, upsample: bool):
    X_train, X_test, y_train, y_test = load_and_split_classification_data()
    if upsample:
        balance="balanced"
        X_train, y_train = features.upsample(X_train, y_train)
    else:
        balance="imbalanced"

    gnb = GaussianNB(var_smoothing=var_smoothing)
    # - Fit and Predict -
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    y_proba = gnb.predict_proba(X_test)[:,1]

    # - Metrics -
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)

    print(f"--- GaussianNB_var{var_smoothing}_{balance} ---")
    print(f"F1 Score: {f1:.6f}")
    print(f"ROC AUC: {roc_auc:.9f}")
    print(f"Accuracy: {acc:.6f}")
    print(f"Precision: {prec:.6f}")
    print(confusion_matrix(y_test, y_pred))


