from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from data import load_data, clean_data, split_data
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
from sklearn.naive_bayes import GaussianNB

def run_naive_bayes(var_smoothing, upsample: bool):
    X_train, X_test, y_train, y_test = load_and_split_encode_classification_data()
    mlflow.set_experiment("NaiveBayes_Baseline")

    if upsample:
        balance="balanced"
        X_train, y_train = features.upsample(X_train, y_train)
    else:
        balance="imbalanced"

    with mlflow.start_run(run_name=f"GaussianNB_var{var_smoothing}_{balance}"):

        gnb = GaussianNB(var_smoothing=var_smoothing)
        mlflow.log_param("model_type", "GaussianNB")
        mlflow.log_param("var_smoothing", var_smoothing)
        mlflow.log_param("upsampled", upsample)
        # - Fit and Predict -
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_test)
        y_proba = gnb.predict_proba(X_test)[:,1]

        # - Metrics -
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(gnb, name="model")

        print(f"--- GaussianNB_var{var_smoothing}_{balance} ---")
        print(f"F1 Score: {f1:.3f}")
        print(f"ROC AUC: {roc_auc:.6f}")
        print(confusion_matrix(y_test, y_pred))

def tune_naive_bayes(var_smoothing: list):
    for value in var_smoothing:
        run_naive_bayes(value, False)
        run_naive_bayes(value, True)

tune_naive_bayes([1e-7])


# --- Decision Tree Classifier ---
from sklearn.tree import DecisionTreeClassifier
def load_and_split_classification_data():
    classificationData = clean_data(load_data())
    classificationData = features.add_highCharges(classificationData)
    return split_data(classificationData, isClassification=True)
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


