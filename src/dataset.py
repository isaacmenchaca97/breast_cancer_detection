from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    try:
        logger.info("Executing dataser.py...")
        # Load Data
        df = pd.read_csv(input_path)
        df.columns = df.columns.str.lower().str.replace(" ", "_")

        df.to_csv(output_path, index=False)
        logger.success("datase.py Done")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    app()
