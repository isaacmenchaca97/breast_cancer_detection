from pathlib import Path
import pickle

from loguru import logger
import pandas as pd
from sklearn.metrics import accuracy_score
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    logger.info("Performing inference for model...")
    model = pickle.load(open(model_path, "rb"))

    x_test = pd.read_csv(features_path)
    y_test = pd.read_csv(predictions_path)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Model Accuracy: {accuracy}")

    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
