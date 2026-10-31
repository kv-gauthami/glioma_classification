"""
preprocess.py
Dataset cleaning: Cleaning the Glioma dataset and creating a stratified train/test split.
Outputs:
  data/train.csv      -> used for training
  test_data.csv       -> held-out test set (repo root, uploaded in the Streamlit app)
Scaling is NOT done here; it lives inside the model pipelines (see train.py).
"""

import os
from sklearn.model_selection import train_test_split
from data_loader import load_data, TARGET_COL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
TRAIN_PATH = os.path.join(ROOT_DIR, "data", "train.csv")
TEST_PATH = os.path.join(ROOT_DIR, "test_data.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 5157   

df = load_data()
print("Rows before cleaning:", len(df))

# 1. Remove exact duplicate rows (2 found in EDA)
df = df.drop_duplicates().reset_index(drop=True)
print("Rows after removing duplicates:", len(df))

# 2. Stratified split keeps the LGG/GBM ratio the same in train and test
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df[TARGET_COL],
    random_state=RANDOM_STATE,
)

# 3. Save (unscaled, with target column included)
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print(f"\nTrain shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")
print("\nTrain class %:\n", (train_df[TARGET_COL].value_counts(normalize=True) * 100).round(2))
print("\nTest class %:\n", (test_df[TARGET_COL].value_counts(normalize=True) * 100).round(2))
print(f"\nSaved: {TRAIN_PATH}\nSaved: {TEST_PATH}")