from pathlib import Path
import pickle

import dagshub
from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import typer

from src.config import MODELS_DIR, N_COMPONENTS, PROCESSED_DATA_DIR, RANDOM_STATE

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    test_features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    test_predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    try:
        logger.info("Training some model...")
        X = pd.read_csv(features_path)
        y = pd.read_csv(labels_path)

        label, counts = np.unique(y, return_counts=True)
        logger.debug(f"label: {label}\ncounts: {counts}")

        counts = max(counts) / counts
        class_weights = dict(zip(label, np.around(counts, 3)))
        logger.debug("=== CLASS WEIGHTS ===")
        logger.debug(class_weights)

        dagshub.init(
            repo_owner="isaacmenchaca97", repo_name="breast_cancer_detection", mlflow=True
        )
        mlflow.sklearn.autolog()
        mlflow.set_experiment("Breast Cancer Classification")

        with mlflow.start_run():
            x_train, x_test, y_train, y_test = train_test_split(
                X, y.to_numpy().ravel(), test_size=0.20, random_state=RANDOM_STATE
            )
            model_pipeline = make_pipeline(
                MinMaxScaler(),
                PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE),
                LogisticRegression(
                    solver="liblinear", class_weight=class_weights, random_state=RANDOM_STATE
                ),
                memory=None,
            )

            model_pipeline.fit(x_train, y_train)
            y_pred = model_pipeline.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("Accuracy", accuracy)

        pickle.dump(model_pipeline, open(model_path, "wb"))
        pd.DataFrame(x_test).to_csv(test_features_path, index=False)
        pd.DataFrame(y_test).to_csv(test_predictions_path, index=False)

        logger.success("Modeling training complete.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    app()
