"""
data_loader.py
This file loads the Glioma Grading (TCGA) dataset: 20 gene-mutation flags + clinical features.
Target: Grade (0 = LGG, 1 = GBM).
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "TCGA_InfoWithGrade.csv")

TARGET_COL = "Grade"
DROP_COLS = ["Race"]   # sensitive attribute, integer-coded categories -> excluded


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the preprocessed TCGA CSV file and return a DataFrame."""
    df = pd.read_csv(path)
    df = df.drop(columns=DROP_COLS)     # dropping sensitive attribute
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """All columns except the target are features."""
    return [c for c in df.columns if c != TARGET_COL]


if __name__ == "__main__":
    df = load_data()
    features = get_feature_cols(df)

    print("Shape:", df.shape)
    print("Number of features:", len(features))
    print("\nColumns:", list(df.columns))
    print("\nFirst rows:\n", df.head())
    print("\nTotal missing values:", df.isnull().sum().sum())
    print("\nClass distribution (0=LGG, 1=GBM):\n", df[TARGET_COL].value_counts())
    print("\nClass proportions (%):\n",
          (df[TARGET_COL].value_counts(normalize=True) * 100).round(2))