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