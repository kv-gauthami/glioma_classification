# Glioma Grade Classification (LGG vs GBM)

**Live app:** https://gliomaclassification-2026ak05157.streamlit.app/
**GitHub repository:** https://github.com/kv-gauthami/glioma_classification

---

## a. Problem Statement

Gliomas are the most common primary tumours of the brain, and they are broadly graded as either **LGG (lower-grade glioma)** or **GBM (glioblastoma multiforme)**. The difference matters a lot in practice: GBM is fast-growing and aggressive with a much poorer prognosis, so the grade directly shapes treatment decisions. Grading is traditionally done from tissue histology and imaging, but molecular markers, especially gene mutations, have become an important part of the picture.

The goal of this project is to predict a patient's glioma grade (LGG or GBM) from a small set of clinical details and gene-mutation flags. I trained and compared five classification models on the same data, evaluated each on six metrics, and built an interactive Streamlit app that loads the trained models and evaluates them on uploaded test data.

---

## b. Dataset Description

- **Source:** UCI Machine Learning Repository, *Glioma Grading Clinical and Mutation Features* (Tasci et al., 2022). The data was built from two NIH cancer genome projects, TCGA-LGG and TCGA-GBM. License: CC BY 4.0.
- **Instances:** 839 patients, with 487 LGG and 352 GBM (roughly a 58/42 split, so mildly imbalanced).
- **Features (23 in the original file):**
  - 3 clinical features: Gender (0 = male, 1 = female), Age_at_diagnosis (in years, continuous), and Race.
  - 20 molecular features: whether each of the 20 most frequently mutated genes in glioma is mutated (1) or not (0). These include IDH1, TP53, ATRX, PTEN, EGFR, CIC, MUC16, PIK3CA, NF1, PIK3R1, FUBP1, RB1, NOTCH1, BCOR, CSMD3, SMARCA4, GRIN2A, IDH2, FAT4 and PDGFRA.
- **Target:** Grade (0 = LGG, 1 = GBM), so this is a binary classification problem.
- **Missing values:** none. I used the preprocessed file `TCGA_InfoWithGrade.csv`, where incomplete records had already been removed. I deliberately avoided the raw file, because it contains a `Primary_Diagnosis` column that essentially gives away the grade.

### Preprocessing

- **Dropped `Race`.** It is a nominal category stored as integers (0–3), so using it as a number would imply a false ordering between groups. It is also a sensitive attribute with heavily unbalanced categories, and I did not want a medical model to pick up spurious associations from it. This leaves **22 features**, still well above the minimum of 12.
- **Removed 2 exact duplicate rows**, leaving 837 patients. This prevents the same record from appearing in both the training and test sets.
- **Stratified 80/20 train/test split** (random seed 5157), giving 669 training patients and 168 test patients (97 LGG, 71 GBM). Stratification keeps the class ratio the same in both sets.
- **Scaling:** Logistic Regression, KNN and SVM are sensitive to feature scale, so each is wrapped in a pipeline with a `StandardScaler`. The scaler is fitted on the training data only and saved inside the model file, so the app applies exactly the same scaling. Decision Tree and Naive Bayes don't need scaling.
- **Hyperparameter tuning:** done with 5-fold stratified cross-validation on the training set only, using MCC as the selection metric. The test set was never used for tuning.

### Key findings from exploratory analysis

- **Age is a strong signal.** GBM patients are noticeably older, with an average age of about 60.7 compared with about 43.9 for LGG.
- **IDH1 is the single strongest marker.** It is mutated in 78% of LGG patients but only 6.5% of GBM patients, which is consistent with what is known about glioma biology.
- **The mutations fall into two groups.** ATRX, CIC, NOTCH1, FUBP1 and IDH2 are much more common in LGG, while PTEN, EGFR, RB1 and PDGFRA are more common in GBM.
- **Several genes are rarely mutated** (under 5% of patients), and a few, such as BCOR and PIK3CA, show almost no difference between the two grades, so they carry little predictive information.

---

## c. GitHub Repository

**Link:** https://github.com/kv-gauthami/glioma_classification

```
glioma_classification/
├── app.py                  # Streamlit app
├── requirements.txt        # pinned dependencies
├── README.md
├── test_data.csv           # held-out test set (upload this in the app)
├── data/                   # raw dataset and training split
├── model/                  # the 5 trained models (.pkl)
├── results/                # metric tables (baseline and tuned) and best hyperparameters
├── figures/                # EDA plots
└── src/
    ├── data_loader.py      # loads the dataset and drops excluded columns
    ├── eda.py              # exploratory analysis and plots
    ├── preprocess.py       # duplicate removal and stratified split
    ├── metrics.py          # computes the 6 evaluation metrics
    ├── tune_model.py       # hyperparameter tuning with cross-validation
    └── train.py            # trains, evaluates and saves all 5 models
```

---

## d. Models Used

All five models were trained on the same training set and evaluated on the same held-out test set of 168 patients. For Precision, Recall and F1, the positive class is **GBM (1)**, since correctly identifying the aggressive tumours is the clinically important case.

For Naive Bayes I used the **Gaussian** variant. Multinomial Naive Bayes assumes count data (like word frequencies), which doesn't fit a dataset made of a continuous age variable and binary mutation flags.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9107 | 0.9348 | 0.8684 | 0.9296 | 0.8980 | 0.8203 |
| Decision Tree | 0.8929 | 0.8956 | 0.8442 | 0.9155 | 0.8784 | 0.7850 |
| KNN | 0.8750 | 0.9193 | 0.8472 | 0.8592 | 0.8531 | 0.7444 |
| Naive Bayes | 0.8512 | 0.9071 | 0.7738 | 0.9155 | 0.8387 | 0.7109 |
| Support Vector Machine | 0.8988 | 0.9286 | 0.8553 | 0.9155 | 0.8844 | 0.7960 |

*Tuned hyperparameters: Logistic Regression C = 0.1; Decision Tree max_depth = 2 (gini); KNN k = 9 with Manhattan distance; SVM (RBF kernel) C = 10, gamma = 0.001. Naive Bayes was left at its defaults.*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | The best model overall: it scores highest on all six metrics (Accuracy, AUC, Precision, Recall, F1 and MCC). It missed only 5 of the 71 GBM patients and made 15 errors in total. Tuning picked a stronger regularisation (C = 0.1), meaning a simpler, smoother boundary generalised better. The fact that a simple linear model does this well suggests the two grades are separated by a roughly linear boundary, driven mainly by age and a few strong markers like IDH1. |
| Decision Tree | This model gained the most from tuning. Without a depth limit it overfit badly (MCC 0.48); limiting it to a depth of just 2 raised MCC to 0.79. A tree that shallow is most likely splitting on IDH1 and then age, which matches the EDA. Its AUC is the lowest of the five, because a depth-2 tree only produces a handful of distinct probability values, so it can't rank patients as finely as the other models. |
| KNN | Performed reasonably well after tuning (k = 9 with Manhattan distance), improving from an MCC of 0.66 to 0.74. Manhattan distance suits this data because, for binary features, it simply counts how many mutation flags differ between two patients. Even so, it missed the most GBM cases (10 of 71), giving it the lowest Recall. With 20 of the 22 features being binary, many patients end up at very similar distances, which makes the nearest neighbours less informative. |
| Naive Bayes | Has high Recall (0.916) but the lowest Precision (0.774): it raised 19 false alarms, nearly double the other models, so it tends to over-predict GBM. This is understandable, as its assumptions don't really hold here. Gaussian NB expects continuous, bell-shaped features, but most features are binary, and it assumes features are independent even though several mutations (such as IDH1, ATRX and TP53) tend to occur together. It still does reasonably well because the strongest signals are clear enough to come through anyway. |
| Support Vector Machine | A very close second to Logistic Regression, with similar Accuracy and AUC and the same Recall as Naive Bayes, but with much better Precision. Interestingly, tuning chose a very small gamma (0.001), which makes the RBF kernel behave almost like a linear one. So the extra flexibility of the RBF kernel wasn't needed, which again points to a mostly linear boundary between the grades. |
| **Overall, winner for your dataset?** | **Logistic Regression.** It is the top model on every one of the six metrics and, most importantly for this problem, misses the fewest aggressive GBM tumours. It is also the simplest and most interpretable of the five models. One caveat: the test set has only 168 patients, so small differences (for example between Logistic Regression and SVM) may not be statistically meaningful, and a cross-validated comparison would give a more robust ranking. |

---

## Streamlit App Features

- Upload a test CSV (the 22 features plus the `Grade` column), with validation and clear error messages for missing columns
- Download the sample `test_data.csv` directly from the app
- Choose any of the five models from a dropdown
- See all six evaluation metrics for the selected model
- Confusion matrix and full classification report
- Per-patient predictions with predicted probability, an option to show only misclassified patients, and a download button
- A comparison tab showing all five models side by side on the uploaded data
- The app only loads the saved models; it never retrains on the uploaded data

---

## How to Run Locally

```bash
pip install -r requirements.txt
python src/preprocess.py      # creates the train/test split
python src/train.py           # trains and saves the 5 models
streamlit run app.py          # launches the app
```

---

## Reference

Tasci, E., Camphausen, K., Krauze, A., & Zhuge, Y. (2022). *Glioma Grading Clinical and Mutation Features* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5R62J