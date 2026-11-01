"""
tune.py
Hyperparameter tuning with 5-fold stratified cross-validation on the train set.
Best parameters are printed and saved to results/best_params.json.
"""

import os
import json
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, matthews_corrcoef

from train import build_models, load_split, TRAIN_PATH, RESULTS_DIR, RANDOM_STATE

# MCC as the selection metric: it balances all four confusion-matrix cells,
# so it is robust to the 58/42 class split.
SCORER = make_scorer(matthews_corrcoef)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Pipeline models need the "clf__" prefix to reach the classifier inside the pipeline.
PARAM_GRIDS = {
    "Logistic Regression": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
    },
    "Decision Tree": {
        "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
        "min_samples_leaf": [1, 5, 10, 20],
        "criterion": ["gini", "entropy"],
    },
    "KNN": {
        "clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 25],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],          # 1 = Manhattan, 2 = Euclidean
    },
    "SVM": {
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", 0.001, 0.01, 0.1, 1],
    },
    # Naive Bayes has no meaningful hyperparameters to tune here -> kept as default.
}


def main():
    X_train, y_train = load_split(TRAIN_PATH)
    models = build_models()
    best_params = {}

    for name, grid in PARAM_GRIDS.items():
        search = GridSearchCV(models[name], grid, scoring=SCORER, cv=CV, n_jobs=-1)
        search.fit(X_train, y_train)

        best_params[name] = search.best_params_
        print(f"\n{name}")
        print(f"  Best params : {search.best_params_}")
        print(f"  CV MCC      : {search.best_score_:.4f}")

    out_path = os.path.join(RESULTS_DIR, "best_params.json")
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    print(f"\nSaved best parameters to {out_path}")


if __name__ == "__main__":
    main()