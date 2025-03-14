# Data preparation
# ========================================================================
import pandas as pd
import numpy as np
import yaml

# MLFlow
# ========================================================================
import mlflow
import mlflow.sklearn

# Sklearn
# ========================================================================
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

params = yaml.safe_load(open("params.yamls"))["train"]


def train():
    df = pd.read_csv(params["data"])
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
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
    mlflow.set_experiment("Breast Cancer Classification")

    with mlflow.start_run():
        model_pipeline = make_pipeline(
            MinMaxScaler(),
            PCA(n_components=params["n_components"]),
            LogisticRegression(
                solver="liblinear", class_weight=class_weights, random_state=params["ramdom_state"]
            ),
        )

        model_pipeline.fit(X_train, y_train)
