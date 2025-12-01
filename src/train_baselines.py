from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
from data import load_clean_split
from utils import get_filepath
import features
import warnings
import logging

# Suppress MLflow signature warnings
warnings.filterwarnings('ignore', category=UserWarning, module='mlflow')
logging.getLogger('mlflow').setLevel(logging.ERROR)


# --- Naive Bayes Baseline ---
def run_naive_bayes(var_smoothing, upsample: bool):
    X_train, X_test, y_train, y_test = load_clean_split(isClassification=True)
    mlflow.set_experiment("NaiveBayes_Baseline")
    model_name = f"GNB_{var_smoothing}_{upsample}"

    if upsample:
        balance="balanced"
        X_train, y_train = features.upsample(X_train, y_train)
    else:
        balance="imbalanced"

    # Apply one-hot encoding
    X_train = features.encode(X_train)

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
        model_path = get_filepath("models", model_name)
        joblib.dump(gnb, model_path)

        print(f"--- GaussianNB_var{var_smoothing}_{balance} ---")
        print(f"CV ROC-AUC: {cv_roc_auc.mean():.3f} ± {cv_roc_auc.std():.3f}")

        return model_name, cv_roc_auc.mean()


def tune_naive_bayes(var_smoothing: list):
    results = []
    for var_smooth in var_smoothing:
        for upsample in [False, True]:
            model_name, cv_score = run_naive_bayes(var_smooth, upsample)
            results.append({'model': model_name, 'cv_score': cv_score})

    best = max(results, key=lambda x: x['cv_score'])
    path = get_filepath("models", best['model'])
    print(f"Best model: {best['model']} with CV ROC-AUC: {best['cv_score']:.3f} path:{path}")



# --- Decision Tree Classifier ---
def run_decision_tree(max_depth, min_samples_split):
    X_train, X_test, y_train, y_test = load_clean_split(isClassification=True)
    mlflow.set_experiment("DecisionTree_Baseline")
    model_name = f"DT_{max_depth}_{min_samples_split}"

    X_train = features.encode(X_train)

    with mlflow.start_run(run_name=f"DecisionTree_maxDepth{max_depth}_MinSplit{min_samples_split}") as run:

        dtc = DecisionTreeClassifier(
            criterion='gini',
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            class_weight='balanced'
        )

        # Log params
        mlflow.log_param("model_type", "DecisionTree")
        mlflow.log_param("criterion", 'gini')
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)

        # ===== CV on training data =====
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1 = cross_val_score(dtc, X_train, y_train, cv=cv, scoring='f1')
        cv_roc_auc = cross_val_score(dtc, X_train, y_train, cv=cv, scoring='roc_auc')

        mlflow.log_metric("cv_f1_mean", cv_f1.mean())
        mlflow.log_metric("cv_f1_std", cv_f1.std())
        mlflow.log_metric("cv_roc_auc_mean", cv_roc_auc.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_roc_auc.std())

        # Fit on full training set
        dtc.fit(X_train, y_train)

        # Save the model
        mlflow.sklearn.log_model(dtc, name="model")
        model_path = get_filepath("models", model_name)
        joblib.dump(dtc, model_path)

        print(f"--- DecisionTree_maxDepth{max_depth}_MinSplit{min_samples_split} ---")
        print(f"CV ROC-AUC: {cv_roc_auc.mean():.3f} ± {cv_roc_auc.std():.3f}")

        return model_name, cv_roc_auc.mean()


def tune_decision_tree(max_depth: list, min_split: list):
    results = []
    for depth in max_depth:
        for split in min_split:
            model_name, cv_score = run_decision_tree(depth, split)
            results.append({'model': model_name, 'cv_score': cv_score})

    best = max(results, key=lambda x: x['cv_score'])
    path = get_filepath("models", best['model'])
    print(f"\nBest model: {best['model']} with CV ROC-AUC: {best['cv_score']:.3f} path:{path}")
    return best


def tune_classifiers():
    tune_naive_bayes([10e-12, 10e-10, 10e-8, 10e-7])
    tune_decision_tree(max_depth=[3, 5, 10, 15, None], min_split=[5, 10, 20])

# Run tuning
if __name__ == "__main__":
    tune_classifiers()

