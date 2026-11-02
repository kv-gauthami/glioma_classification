# Glioma Grade Classification (LGG vs GBM)

**Live Streamlit App:** https://gliomaclassification-2026ak05157.streamlit.app/
**Repository:**  https://github.com/kv-gauthami/glioma_classification/tree/main

---

## a. Problem Statement

Gliomas are the most common primary brain tumours and are graded as **LGG (Lower-Grade Glioma)** or **GBM (Glioblastoma Multiforme)**. GBM is far more aggressive, so the grade directly affects treatment and prognosis.

**Task:** Binary classification: predict the glioma grade (LGG = 0, GBM = 1) of a patient from clinical features and gene-mutation status.

Five classifiers are trained on the same data, evaluated on six metrics, and deployed in an interactive Streamlit app that evaluates the saved models on uploaded test data.

---

## b. Dataset Description

| Property | Details |
|---|---|
| Source | UCI Machine Learning Repository: *Glioma Grading Clinical and Mutation Features* (Tasci et al., 2022), derived from the TCGA-LGG and TCGA-GBM projects |
| File used | `TCGA_InfoWithGrade.csv` (preprocessed version, all numeric) |
| Instances | 839 patients: 487 LGG, 352 GBM (58% / 42%) |
| Original features | 23: 3 clinical (Gender, Age_at_diagnosis, Race) + 20 binary gene-mutation flags |
| Target | `Grade` (0 = LGG, 1 = GBM) |
| Missing values | None |

**Features used (22):**
- **Gender:** 0 = male, 1 = female
- **Age_at_diagnosis:** age in years (continuous)
- **20 gene-mutation flags** (0 = not mutated, 1 = mutated): IDH1, TP53, ATRX, PTEN, EGFR, CIC, MUC16, PIK3CA, NF1, PIK3R1, FUBP1, RB1, NOTCH1, BCOR, CSMD3, SMARCA4, GRIN2A, IDH2, FAT4, PDGFRA

**Preprocessing:**
1. **Dropped `Race`:** a nominal attribute stored as integers (0–3), which would impose a false numeric order. It is also a sensitive attribute.
2. **Removed 2 duplicate rows**, leaving 837 patients.
3. **Stratified 80/20 train/test split** (random_state = 5157): 669 training and 168 test patients (97 LGG, 71 GBM).
4. **Feature scaling:** `StandardScaler` inside the pipelines for Logistic Regression, KNN and SVM, fitted on the training data only. Decision Tree and Naive Bayes use unscaled features.
5. **Hyperparameter tuning:** `GridSearchCV` with 5-fold stratified cross-validation on the training set, using MCC as the selection metric. The test set was not used for tuning.

**Key EDA findings:**
- Mean age at diagnosis: **60.7 years (GBM)** vs **43.9 years (LGG)**.
- **IDH1** is mutated in **78.2%** of LGG vs **6.5%** of GBM patients, the largest difference of any gene.
- More frequent in LGG: ATRX, CIC, NOTCH1, FUBP1, IDH2. More frequent in GBM: PTEN, EGFR, RB1, PDGFRA.

---

## c. GitHub Repository

**Link:** https://github.com/kv-gauthami/glioma_classification

```
glioma_classification/
├── app.py                      # Streamlit app
├── requirements.txt            # pinned dependencies
├── README.md
├── test_data.csv               # test set (168 patients), used in the app
├── .streamlit/config.toml      # app theme
├── data/
│   ├── TCGA_InfoWithGrade.csv  # raw dataset
│   └── train.csv               # training set
├── model/                      # 5 trained models (.pkl)
├── results/                    # baseline and tuned metric tables, best hyperparameters
├── figures/                    # EDA plots
└── src/
    ├── data_loader.py          # load dataset, drop excluded column
    ├── eda.py                  # exploratory analysis and plots
    ├── preprocess.py           # duplicate removal, stratified split
    ├── metrics.py              # computes the 6 evaluation metrics
    ├── tune_model.py           # hyperparameter tuning (5-fold CV)
    └── train.py                # train, evaluate and save all models
```

---

## d. Models Used

Models: Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, Support Vector Machine. All metrics are computed on the held-out test set (168 patients). For Precision, Recall and F1, the positive class is GBM (1). 

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9107 | 0.9348 | 0.8684 | 0.9296 | 0.8980 | 0.8203 |
| Decision Tree | 0.8929 | 0.8956 | 0.8442 | 0.9155 | 0.8784 | 0.7850 |
| KNN | 0.8750 | 0.9193 | 0.8472 | 0.8592 | 0.8531 | 0.7444 |
| Naive Bayes | 0.8512 | 0.9071 | 0.7738 | 0.9155 | 0.8387 | 0.7109 |
| Support Vector Machine | 0.8988 | 0.9286 | 0.8553 | 0.9155 | 0.8844 | 0.7960 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest score on all six metrics (Recall 0.929, MCC 0.820). Fewest total errors (15) and fewest missed GBM cases (5 of 71). Tuning selected stronger regularisation (C = 0.1), raising MCC from 0.793 to 0.820. |
| Decision Tree | Largest gain from tuning: MCC rose from 0.484 (no depth limit, overfitting) to 0.785 with max_depth = 2. Lowest AUC (0.896), because a depth-2 tree has at most 4 leaves and so outputs only a few distinct probability values, which limits ranking ability. |
| KNN | Improved with tuning (MCC 0.657 to 0.744) using k = 9 and Manhattan distance. Lowest Recall (0.859): it missed the most GBM cases (10 of 71). Its errors are the most evenly split between the two classes (11 LGG predicted as GBM, 10 GBM predicted as LGG). |
| Naive Bayes | High Recall (0.915) but lowest Precision (0.774), from 19 false GBM predictions, the most of any model. Lowest Accuracy (0.851) and MCC (0.711). Gaussian NB models each feature as normally distributed within each class, but 20 of the 22 features are binary (0/1), so this assumption does not fit most of the data. |
| Support Vector Machine | Second best overall (Accuracy 0.899, MCC 0.796), with 17 total errors, 2 more than Logistic Regression. Tuning selected a very small gamma (0.001), under which the RBF kernel behaves close to a linear model; this is consistent with its results being close to those of Logistic Regression. |
| **Overall, Winner for your dataset?** | **Logistic Regression:** best on every metric, fewest total errors, and fewest missed GBM cases, which is the most critical error for this problem. Its margin over SVM is small (2 fewer errors out of 168 test patients). |



## Streamlit App

- **Required features:** CSV upload of test data · model selection dropdown · display of all 6 evaluation metrics · confusion matrix and classification report
- **Additional features:** input validation · warning if uploaded rows overlap with the training data · sample test-data download · per-patient predictions with a filter for misclassified patients · comparison of all 5 models with a chart · data preview
- The app loads the saved models from `model/` and **does not retrain** on uploaded data.

## How to Run Locally

```bash
pip install -r requirements.txt
python src/preprocess.py
python src/train.py
streamlit run app.py
```
## Reference

Tasci, E., Camphausen, K., Krauze, A., & Zhuge, Y. (2022). *Glioma Grading Clinical and Mutation Features* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5R62J
