"""
GliomaLab — Interactive Glioma Grade Classification Workbench
Evaluates five pre-trained classifiers (LGG vs GBM) on an uploaded CSV.
No model training or retraining is performed in this application.
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------
# 1. Project configuration
# ---------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from metrics import evaluate_model  # noqa: E402

TARGET_COL = "Grade"
CLASS_LABELS = ["LGG", "GBM"]
MODEL_DIR = os.path.join(ROOT_DIR, "model")
SAMPLE_PATH = os.path.join(ROOT_DIR, "test_data.csv")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl",
}

st.set_page_config(
    page_title="GliomaLab | Classification Workbench",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# 2. Distinctive visual identity — editorial research workbench
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    :root {
        --paper: #F4F1E8;
        --paper-deep: #E9E9DD;
        --ink: #28372D;
        --forest: #263C32;
        --green: #356B50;
        --muted: #73816F;
        --line: #D9DDD0;
        --white: #FFFFFF;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4, p, label {
        font-family: 'DM Sans', sans-serif;
    }

    h2, h3, h4 {
        color: var(--ink) !important;
    }

    .brand-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 12px;
    }

    .brand-name {
        color: var(--forest);
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        font-size: 19px;
        letter-spacing: 1.8px;
    }

    .brand-tag {
        color: var(--muted);
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 1.2px;
        border: 1px solid var(--line);
        padding: 6px 9px;
        border-radius: 3px;
    }

    .hero-panel {
        background: var(--forest);
        padding: 32px 36px;
        border-radius: 5px;
        margin: 8px 0 24px 0;
        border-left: 5px solid #A5B99B;
    }

    .eyebrow {
        color: #B7CBB7;
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero-title {
        color: #F4F1E8;
        font-family: 'Playfair Display', Georgia, serif;
        font-size: clamp(29px, 3.3vw, 43px);
        line-height: 1.16;
        margin: 17px 0 12px 0;
    }

    .hero-description {
        color: #D1D9CE;
        font-size: 14px;
        line-height: 1.65;
        max-width: 760px;
        margin-bottom: 0;
    }

    .section-kicker {
        color: var(--muted);
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .section-heading {
        color: var(--ink);
        font-size: 22px;
        font-weight: 700;
        margin: 0 0 4px 0;
    }

    .section-subtitle {
        color: #758071;
        font-size: 13px;
        margin: 0 0 17px 0;
    }

    .overview-card {
        background: var(--white);
        border: 1px solid #E2E3D8;
        border-radius: 5px;
        padding: 17px 19px;
        min-height: 105px;
    }

    .overview-label {
        color: var(--muted);
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .overview-value {
        color: var(--green);
        font-size: 27px;
        font-weight: 700;
        margin-top: 7px;
    }

    .overview-note {
        color: #879083;
        font-size: 11px;
        margin-top: 1px;
    }

    .research-note {
        background: #E9EBDD;
        border-left: 3px solid #829779;
        padding: 12px 15px;
        color: #53624F;
        font-size: 12px;
        line-height: 1.6;
        border-radius: 2px;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E3D8;
        border-radius: 5px;
        padding: 15px 17px;
    }

    div[data-testid="stMetricLabel"] {
        color: #718071;
        font-size: 12px;
    }

    div[data-testid="stMetricValue"] {
        color: #356B50;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] {
        background: var(--paper-deep);
        border-right: 1px solid var(--line);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: #E8E8DC;
        padding: 5px;
        border-radius: 4px;
        gap: 5px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #526353;
        font-size: 13px;
        border-radius: 3px;
    }

    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #263C32 !important;
    }

    .stDownloadButton button {
        border-radius: 4px;
        border: 1px solid #52745D;
        color: #315D43;
        background: #FFFFFF;
        font-weight: 600;
    }

    .stDownloadButton button:hover {
        background: #E9F0E7;
        border-color: #315D43;
        color: #263C32;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E3D8;
        border-radius: 4px;
    }

    hr {
        border-color: var(--line);
    }

    .footer-note {
        color: #899183;
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 1px;
        text-align: center;
        padding: 25px 0 8px 0;
    }

    /* Hide Streamlit's top header and toolbar */
    [data-testid="stHeader"] {
        display: none;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    [data-testid="stDecoration"] {
        display: none;
    }

    /* Adjust the content spacing after hiding the header */
    .stMainBlockContainer {
        padding-top: 1.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# 3. Load the saved models
# ---------------------------------------------------------------------
@st.cache_resource
def load_models():
    loaded = {}
    for model_name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing model file: {path}")
        loaded[model_name] = joblib.load(path)
    return loaded


try:
    models = load_models()
except Exception as exc:
    st.error(f"Could not load the saved models: {exc}")
    st.stop()

first_model = next(iter(models.values()))
if not hasattr(first_model, "feature_names_in_"):
    st.error(
        "The saved model does not expose feature_names_in_. "
        "Please define FEATURE_COLS manually using the exact training feature order."
    )
    st.stop()

FEATURE_COLS = list(first_model.feature_names_in_)

for model_name, model in models.items():
    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != FEATURE_COLS:
            st.error(
                f"{model_name} was trained with a different feature list/order. "
                "Check the saved models before evaluating them together."
            )
            st.stop()

# ---------------------------------------------------------------------
# 4. Helper functions
# ---------------------------------------------------------------------
def validate_dataset(data):
    """Return an error message if the uploaded CSV is not evaluation-ready."""
    if data.empty:
        return "The uploaded CSV contains no rows."

    required_cols = FEATURE_COLS + [TARGET_COL]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        return f"Missing required columns: {missing_cols}"

    # Convert a copy of the feature columns to numeric for validation.
    try:
        numeric_features = data[FEATURE_COLS].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError):
        return "All model input features must contain numeric values."

    if numeric_features.isna().any().any():
        cols = numeric_features.columns[numeric_features.isna().any()].tolist()
        return f"Missing values found in feature columns: {cols}"

    if not np.isfinite(numeric_features.to_numpy(dtype=float)).all():
        return "Infinite or non-finite values were found in the feature columns."

    if data[TARGET_COL].isna().any():
        return "The Grade column contains missing values."

    # Accept integer-like 0/1 values, but reject unexpected labels.
    try:
        target_numeric = pd.to_numeric(data[TARGET_COL], errors="raise")
    except (ValueError, TypeError):
        return "Grade must be encoded numerically: 0 = LGG and 1 = GBM."

    if not set(target_numeric.unique()).issubset({0, 1}):
        return "Invalid Grade values. Expected only 0 (LGG) and 1 (GBM)."

    if target_numeric.nunique() < 2:
        return (
            "The test dataset must contain both classes (LGG and GBM) "
            "to calculate the full set of evaluation metrics."
        )

    return None


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    fig.patch.set_facecolor("#FFFFFF")
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=sns.light_palette("#356B50", as_cmap=True),
        cbar=False,
        linewidths=1,
        linecolor="#FFFFFF",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        annot_kws={"size": 14, "weight": "bold", "color": "#263C32"},
        ax=ax,
    )
    ax.set_xlabel("Predicted grade", color="#526353", labelpad=10)
    ax.set_ylabel("Actual grade", color="#526353", labelpad=10)
    ax.set_title(title, fontsize=12, fontweight="bold", color="#28372D", pad=14)
    ax.set_facecolor("#FFFFFF")
    fig.tight_layout()
    return fig


def plot_class_distribution(y_true):
    counts = (
        pd.Series(y_true)
        .map({0: "LGG", 1: "GBM"})
        .value_counts()
        .reindex(CLASS_LABELS, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    fig.patch.set_facecolor("#FFFFFF")
    bars = ax.bar(
        counts.index,
        counts.values,
        color=["#91A9BD", "#477A5D"],
        width=0.58,
    )
    ax.bar_label(bars, padding=4, fontsize=10, color="#526353")
    ax.set_ylabel("Number of patients", color="#526353")
    ax.set_xlabel("")
    ax.set_title("Class distribution", fontsize=12, fontweight="bold", color="#28372D")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#D9DDD0")
    ax.tick_params(colors="#647264")
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def get_gbm_probability(model, X):
    """Return P(GBM) when predict_proba is supported and class 1 is present."""
    if not hasattr(model, "predict_proba"):
        return None

    try:
        probabilities = model.predict_proba(X)
        classes = list(model.classes_)
        if 1 not in classes:
            return None
        return probabilities[:, classes.index(1)]
    except (AttributeError, ValueError, TypeError, IndexError):
        return None


def metric_value(value):
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)



# ---------------------------------------------------------------------
# 5. Sidebar
# ---------------------------------------------------------------------
# Initialize dataset-selection variables before entering the sidebar block.
# Streamlit reruns this script from top to bottom on every interaction, so these
# variables must always exist before they are referenced later in the app.
df = None
uploaded_file = None

with st.sidebar:
    st.markdown(
        """
        <div class="brand-name">GLIOMALAB<span style="color:#789477">.</span></div>
        <div class="section-kicker">RESEARCH WORKBENCH</div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("#### 01 / Data source")

    dataset_source = st.radio(
        "Choose evaluation dataset",
        options=[
            "Prepared test dataset",
            "Upload my own CSV",
        ],
        index=0,
    )

    if dataset_source == "Prepared test dataset":

        if os.path.isfile(SAMPLE_PATH):
            try:
                df = pd.read_csv(SAMPLE_PATH)
                st.sidebar.success("Test dataset loaded successfully")
                st.sidebar.caption(
                    f"{len(df)} patient records available"
                )
            except Exception as e:
                st.sidebar.error(f"Could not read test dataset: {e}")

        else:
            st.sidebar.error(
                "test_data.csv not found. "
                "Place it in the project folder."
            )

    # Alternative: user uploads their own dataset
    else:
        uploaded_file = st.file_uploader(
            "Import evaluation dataset",
            type=["csv"],
            help="CSV containing all model features and the Grade target column.",
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success("Custom dataset uploaded")
                st.sidebar.caption(
                    f"{len(df)} patient records available"
                )
            except Exception as e:
                st.sidebar.error(f"Could not read uploaded CSV: {e}")

    # Sample dataset download
    if os.path.isfile(SAMPLE_PATH):
        with open(SAMPLE_PATH, "rb") as sample_file:
            st.download_button(
                "↓  Download sample CSV",
                data=sample_file,
                file_name="test_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.divider()

    st.markdown("#### 02 / Classifier")
    selected_model_name = st.selectbox(
        "Choose algorithm",
        list(MODEL_FILES.keys()),
    )

    st.divider()

    st.markdown("#### 03 / Class definitions")
    st.markdown("**0 — LGG**  \nLower-grade glioma")
    st.markdown("**1 — GBM**  \nGlioblastoma")

    st.divider()
    st.caption("GLIOMALAB · VERSION 1.0")
    st.caption("Evaluation only · No model training")


# ---------------------------------------------------------------------
# 6. Header and project information
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="brand-row">
        <div class="brand-name">GLIOMALAB<span style="color:#789477">.</span></div>
        
    </div>
    <div class="hero-panel">
        <div class="eyebrow">MACHINE LEARNING CLASSIFICATION STUDY</div>
        <div class="hero-title">
            Glioma grade classification
        </div>
        <p class="hero-description">
            An interactive research workbench for evaluating five pre-trained
            classifiers on lower-grade glioma (LGG) and glioblastoma (GBM) data.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("About"):
    info_left, info_right = st.columns(2)
    with info_left:
        st.markdown(
            """
            **Dataset**
            - 839 TCGA glioma patients 
            - UCI Machine Learning Repository
            - Target: `Grade` (0 = LGG, 1 = GBM)
            """
        )
    with info_right:
        st.markdown(
            """
            **Pre-trained classifiers**
            - Logistic Regression
            - Decision Tree
            - K-Nearest Neighbors
            - Gaussian Naive Bayes
            - Support Vector Machine

            """
        )

# ---------------------------------------------------------------------
# 7. Data validation
# ---------------------------------------------------------------------

# Select the correct dataset
if dataset_source == "Prepared test dataset":

    if df is None:
        st.error(
            "Prepared test dataset is unavailable. "
            "Please check whether test_data.csv exists."
        )
        st.stop()

    raw_df = df.copy()

else:

    # The uploaded CSV has already been read into `df` in the sidebar.
    # Do not reference/re-read `uploaded_file` here; using `df` also makes
    # the data flow robust across Streamlit reruns.
    if df is None:
        st.info(
            "Please upload a labelled CSV file using the sidebar "
            "to begin model evaluation."
        )
        st.stop()

    raw_df = df.copy()


# Validate the selected dataset
validation_error = validate_dataset(raw_df)

if validation_error:
    st.error(validation_error)
    st.stop()


# Prepare evaluation data
df = raw_df.copy()

X = df[FEATURE_COLS].apply(pd.to_numeric, errors="raise")

y = pd.to_numeric(
    df[TARGET_COL],
    errors="raise"
).astype(int)


# Dataset statistics
n_patients = len(df)
n_lgg = int((y == 0).sum())
n_gbm = int((y == 1).sum())

# ---------------------------------------------------------------------
# 8. Dataset snapshot
# ---------------------------------------------------------------------
st.markdown('<div class="section-kicker">DATASET / SNAPSHOT</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Evaluation cohort</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-subtitle">A quick overview of the labelled records supplied for evaluation.</p>',
    unsafe_allow_html=True,
)

snapshot_cols = st.columns(4)
snapshot_items = [
    ("PATIENT RECORDS", f"{n_patients:,}", "Uploaded observations"),
    ("LGG CASES", f"{n_lgg:,}", f"{n_lgg / n_patients:.1%} of cohort"),
    ("GBM CASES", f"{n_gbm:,}", f"{n_gbm / n_patients:.1%} of cohort"),
    ("MODEL FEATURES", str(len(FEATURE_COLS)), "Features passed to classifiers"),
]
for col, (label, value, note) in zip(snapshot_cols, snapshot_items):
    with col:
        st.markdown(
            f"""
            <div class="overview-card">
                <div class="overview-label">{label}</div>
                <div class="overview-value">{value}</div>
                <div class="overview-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")
st.success(f"Dataset validated · {n_patients:,} patient records ready for evaluation.")

# ---------------------------------------------------------------------
# 9. Research workspace navigation
# ---------------------------------------------------------------------
page = st.radio(
    "Research workspace",
    [
        "01 · Evaluation",
        "02 · Model comparison",
        "03 · Patient explorer",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()

# ---------------------------------------------------------------------
# 10. Evaluation page
# ---------------------------------------------------------------------
if page == "01 · Evaluation":
    model = models[selected_model_name]
    y_pred = model.predict(X)

    scores = evaluate_model(model, X, y)

    st.markdown('<div class="section-kicker">MODULE 01 / CLASSIFIER PERFORMANCE</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-heading">{selected_model_name}</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-subtitle">Performance metrics calculated on the currently uploaded labelled dataset.</p>',
        unsafe_allow_html=True,
    )

    score_items = list(scores.items())
    for start in range(0, len(score_items), 3):
        metric_cols = st.columns(3)
        for col, (metric_name, value) in zip(metric_cols, score_items[start:start + 3]):
            col.metric(metric_name, metric_value(value))

    st.markdown("")
    st.divider()
    st.markdown("#### Classification diagnostics")

    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown("**Confusion matrix**")
        cm_fig = plot_confusion_matrix(y, y_pred, selected_model_name)
        st.pyplot(cm_fig, use_container_width=True)
        plt.close(cm_fig)
        st.caption("Rows = actual class · Columns = predicted class")

    with right:
        st.markdown("**Classification report**")
        report = classification_report(
            y,
            y_pred,
            labels=[0, 1],
            target_names=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.round(3), use_container_width=True)
        st.caption("Per-class precision, recall and F1-score.")

    st.markdown("")
    st.markdown("#### Error profile")
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    err1, err2, err3, err4 = st.columns(4)
    err1.metric("Correct LGG (TN)", f"{tn:,}")
    err2.metric("LGG predicted as GBM (FP)", f"{fp:,}")
    err3.metric("GBM predicted as LGG (FN)", f"{fn:,}")
    err4.metric("Correct GBM (TP)", f"{tp:,}")

    # st.markdown(
    #     """
    #     <div class="research-note">
    #         <strong>Interpretation note:</strong> a false negative (FN) is a GBM
    #         case predicted as LGG; a false positive (FP) is an LGG case predicted
    #         as GBM. Review both error types rather than relying on accuracy alone.
    #     </div>
    #     """,
    #     unsafe_allow_html=True,
    # )

    st.markdown("")
    st.divider()
    st.markdown("#### Patient-level predictions")

    pred_df = df.copy()
    pred_df["Actual"] = y.map({0: "LGG", 1: "GBM"})
    pred_df["Predicted"] = pd.Series(y_pred, index=df.index).map({0: "LGG", 1: "GBM"})
    pred_df["Correct"] = pred_df["Actual"] == pred_df["Predicted"]

    gbm_probability = get_gbm_probability(model, X)
    if gbm_probability is not None:
        pred_df["P(GBM)"] = np.round(gbm_probability, 4)

    filter_a, filter_b = st.columns(2)
    with filter_a:
        error_filter = st.selectbox(
            "Prediction status",
            ["All patients", "Misclassified only", "Correctly classified only"],
        )
    with filter_b:
        actual_filter = st.selectbox(
            "Actual grade",
            ["All grades", "LGG", "GBM"],
        )

    patient_view = pred_df.copy()
    if error_filter == "Misclassified only":
        patient_view = patient_view[~patient_view["Correct"]]
    elif error_filter == "Correctly classified only":
        patient_view = patient_view[patient_view["Correct"]]

    if actual_filter != "All grades":
        patient_view = patient_view[patient_view["Actual"] == actual_filter]

    preferred_cols = [
        "Age_at_diagnosis",
        "Actual",
        "Predicted",
        "P(GBM)",
        "Correct",
    ]
    display_cols = [col for col in preferred_cols if col in patient_view.columns]
    if not display_cols:
        display_cols = ["Actual", "Predicted", "Correct"]

    st.caption(f"Showing {len(patient_view):,} of {len(pred_df):,} patients.")
    st.dataframe(
        patient_view[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "↓  Download patient predictions",
        data=pred_df.to_csv(index=False).encode("utf-8"),
        file_name=f"gliomalab_predictions_{selected_model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------
# 11. Model comparison page
# ---------------------------------------------------------------------
elif page == "02 · Model comparison":
    st.markdown('<div class="section-kicker">MODULE 02 / MODEL COMPARISON</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Compare the five classifiers</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-subtitle">Every classifier is evaluated against the same uploaded test records.</p>',
        unsafe_allow_html=True,
    )

    comparison_rows = []
    for name, model in models.items():
        model_scores = evaluate_model(model, X, y)
        comparison_rows.append({"Model": name, **model_scores})

    comparison_df = pd.DataFrame(comparison_rows).set_index("Model")
    comparison_df = comparison_df.apply(pd.to_numeric, errors="coerce")

    st.markdown("#### Results table")
    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
    )

    if "MCC" in comparison_df.columns and comparison_df["MCC"].notna().any():
        best_name = comparison_df["MCC"].idxmax()
        best_mcc = comparison_df.loc[best_name, "MCC"]
        st.markdown(
            f"""
            <div class="research-note">
                <strong>Top model by MCC:</strong> {best_name}
                &nbsp;·&nbsp; MCC = {best_mcc:.4f}<br>
                MCC ranges from -1 to +1. Matthews Correlation Coefficient,
                considers TP, TN, FP, and FN. A higher MCC indicates stronger agreement
                between predicted and actual classes.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # st.download_button(
    #     "↓  Download model comparison",
    #     data=comparison_df.to_csv().encode("utf-8"),
    #     file_name="gliomalab_model_comparison.csv",
    #     mime="text/csv",
    # )

    st.markdown("")
    st.divider()
    st.markdown("#### Metric explorer")

    available_metrics = [
        col for col in comparison_df.columns
        if comparison_df[col].notna().any()
    ]
    default_metrics = [
        metric for metric in ["Accuracy", "Precision", "Recall", "F1", "AUC", "MCC"]
        if metric in available_metrics
    ][:6]

    chosen_metrics = st.multiselect(
        "Choose metrics for the comparison plot",
        options=available_metrics,
        default=default_metrics or available_metrics[:2],
    )

    if chosen_metrics:
        # Subtle GliomaLab colour palette
        theme_colours = [
            "#527968",  # Forest green
            "#789482",  # Sage green
            "#A2B5A3",  # Soft eucalyptus
            "#C1B89A",  # Muted olive
            "#D1A77E",  # Warm sand
            "#9A8575",  # Muted taupe
        ]

        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")

        comparison_df[chosen_metrics].plot(
            kind="bar",
            ax=ax,
            width=0.76,
            color=theme_colours[:len(chosen_metrics)],
            edgecolor="none",
        )

        ax.set_title(
            "Classifier performance across selected metrics",
            fontsize=13,
            fontweight="bold",
            color="#28372D",
            pad=14,
        )
        ax.set_ylabel("Metric score", color="#526353")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0, colors="#526353")
        ax.tick_params(axis="y", colors="#526353")
        ax.grid(axis="y", color="#D9DDD0", alpha=0.45, linewidth=0.8)
        ax.set_axisbelow(True)

        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#D9DDD0")

        ax.legend(
            frameon=False,
            ncol=min(6, len(chosen_metrics)),
            labelcolor="#526353",
        )

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    else:
        st.info("Choose at least one metric to display the chart.")


# ---------------------------------------------------------------------
# 12. Patient explorer page
# ---------------------------------------------------------------------
elif page == "03 · Patient explorer":
    st.markdown('<div class="section-kicker">MODULE 03 / COHORT INSPECTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Explore the uploaded cohort</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-subtitle">Overview of the patient data</p>',
        unsafe_allow_html=True,
    )

    chart_col, info_col = st.columns([1.05, 0.95])
    with chart_col:
        st.markdown("**Distribution of target classes**")
        distribution_fig = plot_class_distribution(y)
        st.pyplot(distribution_fig, use_container_width=True)
        plt.close(distribution_fig)

    with info_col:
        st.markdown("**Cohort summary**")
        summary = pd.DataFrame(
            {
                "Grade": ["LGG", "GBM"],
                "Encoded label": [0, 1],
                "Patients": [n_lgg, n_gbm],
                "Percentage": [
                    f"{n_lgg / n_patients:.1%}",
                    f"{n_gbm / n_patients:.1%}",
                ],
            }
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown(
            f"""
            <div class="research-note">
                <strong>Feature count:</strong> {len(FEATURE_COLS)}<br>
                <strong>Target variable:</strong> {TARGET_COL}<br>
                <strong>Records with missing required values:</strong> 0
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("#### Uploaded data preview")
    st.dataframe(df.head(25), use_container_width=True, hide_index=True)
    st.caption(
        f"Showing the first {min(25, n_patients)} of {n_patients:,} records. "
    )


# ---------------------------------------------------------------------
# 13. Footer
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-note">
        GLIOMALAB · GLIOMA GRADE CLASSIFICATION · RESEARCH WORKBENCH<br>
        LGG vs GBM &nbsp;|&nbsp; FIVE PRE-TRAINED CLASSIFIERS
    </div>
    """,
    unsafe_allow_html=True,
)
