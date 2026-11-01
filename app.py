"""
app.py
Streamlit app: Glioma grade classification (LGG vs GBM).
Loads the 5 pre-trained models from model/ and evaluates them on an uploaded
test CSV. No retraining happens here.
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

# Make src/ importable for the app to reuse the metrics
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "src"))
from metrics import evaluate_model  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Config & constants
# ---------------------------------------------------------------------------
TARGET_COL = "Grade"
CLASS_LABELS = ["LGG", "GBM"]          # 0 = LGG, 1 = GBM
MODEL_DIR = os.path.join(ROOT_DIR, "model")
SAMPLE_PATH = os.path.join(ROOT_DIR, "test_data.csv")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl",
}

st.set_page_config(page_title="Glioma Grade Classifier", page_icon="🧠", layout="wide")

with st.expander("ℹ️ About this project"):
    st.markdown(
        "- **Data:** 839 TCGA glioma patients (UCI ML Repository)\n"
        "- **Features:** age, gender + 20 gene-mutation flags\n"
        "- **Models:** Logistic Regression, Decision Tree, KNN, "
        "Gaussian Naive Bayes, SVM\n"
        "- Hyperparameters tuned with 5-fold cross-validation\n"
    )


@st.cache_resource
def load_models():
    """Load all saved models once and cache them."""
    return {name: joblib.load(os.path.join(MODEL_DIR, fname))
            for name, fname in MODEL_FILES.items()}


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("Actual grade")
    ax.set_title(title)
    fig.tight_layout()
    return fig


models = load_models()
FEATURE_COLS = list(next(iter(models.values())).feature_names_in_)

# ---------------------------------------------------------------------------
# 2. Header & sidebar
# ---------------------------------------------------------------------------
st.title("Glioma Grade Classifier")
st.markdown(
    "Predicts whether a brain glioma is **LGG** (lower-grade glioma) or "
    "**GBM** (glioblastoma) from patient age, gender and 20 gene-mutation flags "
    "(TCGA data, UCI repository). Models are pre-trained; this app only evaluates them."
)

with st.sidebar:
    st.header("1. Upload test data")
    uploaded = st.file_uploader("CSV with the 22 features + 'Grade' column", type="csv")

    with open(SAMPLE_PATH, "rb") as f:
        st.download_button("Download sample test_data.csv", f,
                           file_name="test_data.csv", mime="text/csv")

    st.header("2. Choose a model")
    model_name = st.selectbox("Classifier", list(MODEL_FILES.keys()))

    st.divider()
    st.caption("Positive class = GBM (1). Precision/Recall/F1 are reported for GBM.")

# ---------------------------------------------------------------------------
# 3. Validate the uploaded file
# ---------------------------------------------------------------------------
if uploaded is None:
    st.info("Upload a test CSV from the sidebar to begin "
            "(or download the sample file first).")
    st.stop()

df = pd.read_csv(uploaded)

missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
if missing:
    st.error(f"The uploaded file is missing required columns: {missing}")
    st.stop()

X = df[FEATURE_COLS]            # same column order as during training
y = df[TARGET_COL]

if y.nunique() < 2:
    st.error("The test data must contain both classes (LGG and GBM) to compute AUC and MCC.")
    st.stop()

st.success(f"Loaded {len(df)} patients "
           f"({(y == 0).sum()} LGG, {(y == 1).sum()} GBM).")

tab_single, tab_compare, tab_data = st.tabs(
    ["📊 Selected model", "🏆 Compare all models", "🔍 Data preview"]
)

# ---------------------------------------------------------------------------
# 4. Selected model: metrics, confusion matrix, classification report
# ---------------------------------------------------------------------------
with tab_single:
    model = models[model_name]
    scores = evaluate_model(model, X, y)
    y_pred = model.predict(X)

    st.subheader(f"{model_name}: evaluation metrics")
    cols = st.columns(6)
    for col, (metric, value) in zip(cols, scores.items()):
        col.metric(metric, f"{value:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        fig = plot_confusion_matrix(y, y_pred, model_name)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, y_pred, labels=[0, 1],
                                       target_names=CLASS_LABELS,
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), width='stretch')

    # ---- Per-patient predictions ----
    pred_df = df.copy()
    pred_df["Predicted"] = [CLASS_LABELS[p] for p in y_pred]
    pred_df["Actual"] = [CLASS_LABELS[a] for a in y]
    pred_df["P(GBM)"] = model.predict_proba(X)[:, 1].round(3)

    with st.expander("Per-patient predictions"):
        only_errors = st.checkbox("Show only misclassified patients")
        view = pred_df[pred_df["Predicted"] != pred_df["Actual"]] if only_errors else pred_df
        st.caption(f"Showing {len(view)} of {len(pred_df)} patients")
        st.dataframe(view[["Age_at_diagnosis", "Actual", "Predicted", "P(GBM)"]],
                     width='stretch')
        st.download_button("Download predictions",
                           pred_df.to_csv(index=False).encode(),
                           file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
                           mime="text/csv")


# ---------------------------------------------------------------------------
# 5. Compare all five models on the same uploaded data
# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("All models on the uploaded test data")
    rows = [{"Model": name, **evaluate_model(m, X, y)} for name, m in models.items()]
    cmp_df = pd.DataFrame(rows).set_index("Model").round(4)
    # st.dataframe(cmp_df, width='stretch')
    st.dataframe(
    cmp_df,
    width='stretch',
    column_config={
        m: st.column_config.ProgressColumn(m, min_value=0.0, max_value=1.0, format="%.3f")
        for m in cmp_df.columns
    },
    )

    best = cmp_df["MCC"].idxmax()
    st.markdown(f"**Best model by MCC:** {best} ({cmp_df.loc[best, 'MCC']:.3f})")

    fig, ax = plt.subplots(figsize=(10, 4))
    cmp_df.plot(kind="bar", ax=ax, rot=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", ncol=6, fontsize=8)
    ax.set_title("Metric comparison across models")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# 6. Data preview
# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("Uploaded data")
    st.dataframe(df.head(20), width='stretch')
    st.caption(f"Showing first 20 of {len(df)} rows · {len(FEATURE_COLS)} features used by the models.")