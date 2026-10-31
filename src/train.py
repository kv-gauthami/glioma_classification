"""
train.py
Trains the classifiers on data/train.csv, evaluates on test_data.csv,
saves each model to model/ and the comparison table to results/.
"""

import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


from data_loader import TARGET_COL
from metrics import evaluate_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
TRAIN_PATH = os.path.join(ROOT_DIR, "data", "train.csv")
TEST_PATH = os.path.join(ROOT_DIR, "test_data.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

RANDOM_STATE = 5157   

def load_split(path):
    """Read a CSV and split it into features (X) and target (y)."""
    df = pd.read_csv(path)
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL]


def build_models():
    """Return the dict of models to train"""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),

        
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),

        "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),

        "Naive Bayes": GaussianNB(),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, y_train = load_split(TRAIN_PATH)
    X_test, y_test = load_split(TEST_PATH)

    results = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        scores = evaluate_model(model, X_test, y_test)
        results.append({"ML Model Name": name, **scores})

        filename = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, os.path.join(MODEL_DIR, filename))
        print(f"Trained and saved: {name}")

    results_df = pd.DataFrame(results).round(4)
    results_df.to_csv(os.path.join(RESULTS_DIR, "models_comparison.csv"), index=False)
    print("\n", results_df.to_string(index=False))


if __name__ == "__main__":
    main()

