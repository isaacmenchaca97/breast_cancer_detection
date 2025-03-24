# Data preparation
# ========================================================================
import pandas as pd
from scipy import stats
import yaml
import os

from sklearn.datasets import load_breast_cancer


params = yaml.safe_load(open("params.yaml"))["preprocess"]


def preprocess(output_path):
    # Load Data
    data = load_breast_cancer()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df["target"] = data.target
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # ANOVA Statistic
    num_cols = list(df.select_dtypes("float64").columns)
    unrelated_num_cols = []
    categorical_col = "target"

    for i in num_cols:
        # Perform Kruskal-Wallis test
        grouped_data = [
            df[i][df[categorical_col] == category]
            for category in df[categorical_col].unique()
        ]
        statistic, p_value = stats.f_oneway(*grouped_data)

        # Set the significance level (alpha)
        alpha = 0.05

        # Print the results with appropriate text color
        if p_value < alpha:
            print(f"ANOVA statistic: {round(statistic, 2)}")
            print(f"p-value: {p_value:.2e}")
            print(
                "\033[32m"
                + f"Reject the null hypothesis: There is a significant relationship between {i} and {categorical_col}"
            )
            print("\033[0m")  # Reset text color to default
        else:
            print(f"ANOVA statistic: {round(statistic, 2)}")
            print(f"p-value: {p_value:.2e}")
            print(
                "\033[31m"
                + f"No significant relationship between {i} and {categorical_col}"
            )
            print("\033[0m")  # Reset text color to default
            unrelated_num_cols.append(i)

    df = df.drop(labels=unrelated_num_cols, axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    preprocess(params["output"])
