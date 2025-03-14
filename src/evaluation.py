# Data preparation
# ========================================================================
import pandas as pd
import yaml
import dagshub
import pickle

# MLFlow
# ========================================================================
import mlflow

# Sklearn
# ========================================================================
from sklearn.metrics import accuracy_score

params = yaml.safe_load(open("params.yaml"))["train"]


def evaluate():
    df = pd.read_csv(params["data"])
    X = df.drop(columns=["target"])
    y = df["target"]

    mlflow.set_tracking_uri("https://dagshub.com/isaacmenchaca97/breast_cancer_detection.mlflow")
    dagshub.init(repo_owner='isaacmenchaca97', repo_name='breast_cancer_detection', mlflow=True)

    # load the model from the disk
    model = pickle.load(open(params["model"], 'rb'))

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    # log metrics to MLFLOW
    mlflow.log_metric("accuracy", accuracy)
    print(f"Model accuracy:{accuracy}")


if __name__ == "__main__":
    evaluate()
