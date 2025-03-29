from pathlib import Path

from loguru import logger
import pandas as pd
from scipy import stats
import typer

from src.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    feature_output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    label_output_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    # -----------------------------------------
):
    try:
        logger.info("Generating features from dataset...")

        df = pd.read_csv(input_path)

        # ANOVA Statistic
        num_cols = list(df.select_dtypes("float64").columns)
        unrelated_num_cols = []
        categorical_col = "target"

        for i in num_cols:
            # Perform Kruskal-Wallis test
            grouped_data = [
                df[i][df[categorical_col] == category] for category in df[categorical_col].unique()
            ]
            statistic, p_value = stats.f_oneway(*grouped_data)

            # Set the significance level (alpha)
            alpha = 0.05

            # logger.debug the results with appropriate text color
            if p_value < alpha:
                logger.debug(f"ANOVA statistic: {round(statistic, 2)}")
                logger.debug(f"p-value: {p_value:.2e}")
                logger.debug(
                    "\033[32m"
                    + f"Reject the null hypothesis: There is a significant relationship between {i} and {categorical_col}"
                )
                logger.debug("\033[0m")  # Reset text color to default
            else:
                logger.debug(f"ANOVA statistic: {round(statistic, 2)}")
                logger.debug(f"p-value: {p_value:.2e}")
                logger.debug(
                    "\033[31m" + f"No significant relationship between {i} and {categorical_col}"
                )
                logger.debug("\033[0m")  # Reset text color to default
                unrelated_num_cols.append(i)

        df = df.drop(labels=unrelated_num_cols, axis=1)
        df.drop(columns=["target"]).to_csv(feature_output_path, index=False)
        df["target"].to_csv(label_output_path, index=False)

        logger.success("Features generation complete.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    app()
