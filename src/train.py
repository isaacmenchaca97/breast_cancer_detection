# Data preparation
# ========================================================================
import pandas as pd
import numpy as np
import yaml
import dagshub

# MLFlow
# ========================================================================
import mlflow
import mlflow.sklearn

# Sklearn
# ========================================================================
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Model Performance Evaluators
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import os
import pickle


params = yaml.safe_load(open("params.yaml"))["train"]


def train():
    df = pd.read_csv(params["data"])
    X = df.drop(columns=["target"])
    y = df["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=params["ramdom_state"]
    )

    label, counts = np.unique(y, return_counts=True)
    print(f"label: {label}\ncounts: {counts}")
    # compute the class weights
    counts = max(counts) / counts
    class_weights = dict(zip(label, np.around(counts, 3)))
    print("=== CLASS WEIGHTS ===")
    print(class_weights)

    mlflow.set_tracking_uri("https://dagshub.com/isaacmenchaca97/breast_cancer_detection.mlflow")
    dagshub.init(repo_owner='isaacmenchaca97', repo_name='breast_cancer_detection', mlflow=True)
    mlflow.set_experiment("Breast Cancer Classification")

    with mlflow.start_run():
        model_pipeline = make_pipeline(
            MinMaxScaler(),
            PCA(n_components=params["n_components"]),
            LogisticRegression(
                solver="liblinear", class_weight=class_weights, random_state=params["ramdom_state"]
            ),
            memory=None
        )

        model_pipeline.fit(x_train, y_train)
        # making the predictions
        y_pred = model_pipeline.predict(x_test)

        # metrics
        confusionmatrix = np.around(confusion_matrix(y_test, y_pred, normalize="true"), 3)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        # Register parameters
        mlflow.log_param("n_components", params["n_components"])
        mlflow.log_param("solver", "liblinear")

        # Register metrics
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1 Score", f1)
        mlflow.log_metric("ROC AUC", roc_auc)

        mlflow.log_text(str(confusionmatrix), "confusion_matrix.txt")
        mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")

        # Register model
        mlflow.sklearn.log_model(
            model_pipeline, "model_cancer_classification", input_example=x_train.iloc[:2]
        )
        # create the directory to save the model
        os.makedirs(os.path.dirname(params["model"]), exist_ok=True)

        filename = params["model"]
        pickle.dump(model_pipeline, open(filename, 'wb'))

    print("train done")


if __name__ == "__main__":
    train()
