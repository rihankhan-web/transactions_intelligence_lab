
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "transactions_synthetic.csv"
MODEL = ROOT / "models" / "best_model.joblib"

st.set_page_config(
    page_title="Transaction Intelligence Lab",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- PREMIUM DARK THEME ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #080d1b 0%, #10172b 55%, #11132a 100%);
    color: #f4f6ff;
}
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: #f4f6ff !important;
}
p, label, .stCaption {
    color: #b8c3df;
}
[data-testid="stSidebar"] {
    background: #0b1120;
    border-right: 1px solid rgba(255,255,255,.10);
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #17223b, #111a2e);
    border: 1px solid rgba(139,114,255,.32);
    border-radius: 16px;
    padding: 20px;
}
[data-testid="stMetricLabel"] {
    color: #b8c3df;
}
[data-testid="stMetricValue"] {
    color: #53e6d5;
}
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(90deg, #8065ff, #36cfc1);
    color: #080d1b;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    min-height: 44px;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    border: 1px solid #ffffff;
    color: #080d1b;
}
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 12px;
}
hr {
    border-color: rgba(255,255,255,.13);
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data["merchant_category"] = data["merchant_category"].astype(str)
    return data

if not DATA.exists():
    st.error(f"Dataset not found: {DATA}")
    st.stop()

df = load_data(str(DATA))

required = {
    "amount_inr", "merchant_category", "is_fraud",
    "hour", "is_weekend", "new_device",
    "location_mismatch", "failed_attempts"
}
missing = required - set(df.columns)

if missing:
    st.error(f"Dataset mein required columns missing hain: {', '.join(sorted(missing))}")
    st.stop()

# ---------- HEADER ----------
st.markdown(
    '<div style="color:#53e6d5;font-weight:700;letter-spacing:3px;'
    'font-size:12px;">AI / MACHINE LEARNING • PROJECT DASHBOARD</div>',
    unsafe_allow_html=True,
)
st.title("🔎 Transaction Intelligence Lab")
st.markdown(
    "Explore synthetic transaction patterns, inspect dataset labels, "
    "and test a sample transaction against the trained model."
)
st.caption("B.Tech CSE (AI/ML) • Academic portfolio project • Synthetic data only")

st.divider()

# ---------- SIDEBAR FILTERS ----------
with st.sidebar:
    st.markdown("## ⚙️ Dashboard Controls")
    st.caption("Filter the transaction dataset")

    all_categories = sorted(df["merchant_category"].unique().tolist())
    categories = st.multiselect(
        "Merchant category",
        all_categories,
        default=all_categories,
    )

    max_amount = max(1, int(df["amount_inr"].max()))
    upper_default = min(max_amount, int(df["amount_inr"].quantile(.99)))
    amount_range = st.slider(
        "Transaction amount (₹)",
        min_value=0,
        max_value=max_amount,
        value=(0, max(0, upper_default)),
        step=max(1, min(500, max_amount)),
    )

    show_label = st.selectbox(
        "Dataset label",
        ["All", "Normal-labelled", "Fraud-labelled"],
    )

    st.divider()
    st.caption("Tip: Reset filters by selecting all categories and the full amount range.")

# ---------- FILTER DATA ----------
view = df[df["merchant_category"].isin(categories)].copy()
view = view[view["amount_inr"].between(*amount_range)]

if show_label == "Normal-labelled":
    view = view[view["is_fraud"] == 0]
elif show_label == "Fraud-labelled":
    view = view[view["is_fraud"] == 1]

# ---------- KPI CARDS ----------
total = len(view)
total_value = view["amount_inr"].sum()
label_rate = view["is_fraud"].mean() * 100 if total else 0
avg_amount = view["amount_inr"].mean() if total else 0
positive_count = int(view["is_fraud"].sum()) if total else 0

st.markdown("### 📊 Overview")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Transactions", f"{total:,}", help="Rows matching your filters")
k2.metric("Transaction Value", f"₹{total_value:,.0f}")
k3.metric("Positive Labels", f"{positive_count:,}")
k4.metric("Average Amount", f"₹{avg_amount:,.0f}")

st.caption(f"Positive-label rate in filtered data: {label_rate:.2f}%")
st.divider()

# ---------- ANALYTICS ----------
st.markdown("### 📈 Transaction Analytics")
tab1, tab2, tab3 = st.tabs(
    ["Category Analysis", "Label Analysis", "Transaction Data"]
)

with tab1:
    if view.empty:
        st.info("No transactions match these filters. Adjust the sidebar.")
    else:
        category_counts = (
            view["merchant_category"]
            .value_counts()
            .rename_axis("Category")
            .rename("Transactions")
        )
        st.subheader("Transactions by Merchant Category")
        st.bar_chart(category_counts, color="#8b72ff")

        category_amount = (
            view.groupby("merchant_category")["amount_inr"]
            .sum()
            .sort_values(ascending=False)
            .rename("Total Amount (₹)")
        )
        st.subheader("Total Transaction Value by Category")
        st.bar_chart(category_amount, color="#48e3d2")

with tab2:
    if view.empty:
        st.info("No data available for the selected filters.")
    else:
        label_summary = (
            view.groupby("merchant_category")["is_fraud"]
            .agg(["mean", "count"])
            .reset_index()
        )
        label_summary["Label Rate (%)"] = label_summary["mean"] * 100
        label_summary = label_summary.set_index("merchant_category")

        st.subheader("Positive-Label Rate by Category")
        st.bar_chart(label_summary["Label Rate (%)"], color="#ff9f68")

        st.subheader("Label Distribution")
        distribution = (
            view["is_fraud"]
            .map({0: "Normal-labelled", 1: "Fraud-labelled"})
            .value_counts()
            .rename("Transactions")
        )
        st.bar_chart(distribution, color="#48e3d2")

        st.caption(
            "These are simulated dataset labels, not verified real-world fraud cases."
        )

with tab3:
    st.subheader("Transaction Records")
    st.caption(f"Showing up to 200 of {total:,} filtered records.")
    st.dataframe(
        view.head(200),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered CSV",
        data=csv_data,
        file_name="filtered_transactions.csv",
        mime="text/csv",
    )

st.divider()

# ---------- MODEL PREDICTION ----------
st.markdown("### 🧠 Test a Synthetic Transaction")
st.write(
    "Enter sample transaction details to view the score produced by your trained model."
)

with st.form("predict_form"):
    a, b, c = st.columns(3)

    amount = a.number_input(
        "Amount (₹)", min_value=0.0, value=1500.0, step=100.0
    )
    hour = b.slider("Hour of day", 0, 23, 21)
    merchant = c.selectbox("Merchant category", all_categories)

    d, e, f = st.columns(3)
    weekend = d.selectbox(
        "Weekend?", [0, 1], format_func=lambda x: "Yes" if x else "No"
    )
    new_device = e.selectbox(
        "New device?", [0, 1], format_func=lambda x: "Yes" if x else "No"
    )
    mismatch = f.selectbox(
        "Location mismatch?", [0, 1],
        format_func=lambda x: "Yes" if x else "No"
    )
    failed = st.slider("Failed attempts", 0, 7, 0)

    submitted = st.form_submit_button(
        "✨ Score Example", use_container_width=True
    )

if submitted:
    if not MODEL.exists():
        st.error("Model file missing. Run `python src/train.py` first.")
    else:
        try:
            model = joblib.load(MODEL)
            sample = pd.DataFrame([{
                "amount_inr": amount,
                "hour": hour,
                "merchant_category": merchant,
                "is_weekend": weekend,
                "new_device": new_device,
                "location_mismatch": mismatch,
                "failed_attempts": failed,
            }])

            score = float(model.predict_proba(sample)[0, 1])

            st.markdown("#### Model Output")
            p1, p2 = st.columns(2)
            p1.metric("Synthetic Model Score", f"{score * 100:.1f}%")
            p2.metric(
                "Illustrative Score Band",
                "Higher" if score >= 0.5 else "Lower",
            )
            st.progress(max(0.0, min(1.0, score)))
            st.info(
                "This is an uncalibrated model score on synthetic data, "
                "not a verified real-world fraud probability. Do not use it "
                "to approve, block, or investigate real transactions."
            )
        except Exception as exc:
            st.error(f"Prediction error: {exc}")

st.divider()
st.caption(
    "⚠️ Academic demo only • Labels are simulated • "
    "Not a real fraud detection or financial security product."
)



# ---------- ML MODEL PERFORMANCE ----------
st.divider()
st.header("🧠 ML Model Performance")

report_dir = ROOT / "reports"
metrics_file = report_dir / "model_comparison.csv"

if metrics_file.exists():
    metrics_df = pd.read_csv(metrics_file)

    st.subheader("Model Comparison")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    numeric_cols = [
        col for col in
        ["accuracy", "precision", "recall", "f1", "roc_auc"]
        if col in metrics_df.columns
    ]

    if numeric_cols and "model" in metrics_df.columns:
        chart_df = metrics_df.set_index("model")[numeric_cols]
        st.bar_chart(chart_df)

else: 
    st.info(
        "Model comparison report nahi mila. "
       " Reports folder mein available files check karo."
    )



# ---------- MODEL PERFORMANCE DASHBOARD ----------
st.divider()
st.header("🧠 ML Model Performance")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import numpy as np

@st.cache_data
def load_training_data(path):
    return pd.read_csv(path)

if st.button("🔬 Evaluate Model"):
    try:
        data = load_training_data(str(DATA))

        features = [
            "amount_inr",
            "hour",
            "merchant_category",
            "is_weekend",
            "new_device",
            "location_mismatch",
            "failed_attempts",
        ]
        X = data[features]
        y = data["is_fraud"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        if not MODEL.exists():
            st.error("Model file nahi mila. Terminal mein python src/train.py chalao.")
        else:
            model = joblib.load(MODEL)

            # Saved model ko independent test data par evaluate karo.
            # Model ki training mein istemal hua test split alag ho sakta hai.
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(
                y_test, predictions, zero_division=0
            )
            recall = recall_score(
                y_test, predictions, zero_division=0
            )
            f1 = f1_score(y_test, predictions, zero_division=0)
            auc = (
                roc_auc_score(y_test, probabilities)
                if y_test.nunique() == 2 else 0
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{accuracy * 100:.2f}%")
            m2.metric("Precision", f"{precision * 100:.2f}%")
            m3.metric("Recall", f"{recall * 100:.2f}%")

            m4, m5 = st.columns(2)
            m4.metric("F1 Score", f"{f1:.3f}")
            m5.metric("ROC-AUC", f"{auc:.3f}")

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, predictions, labels=[0, 1])
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Normal", "Actual Positive"],
                columns=["Predicted Normal", "Predicted Positive"],
            )
            st.dataframe(cm_df, use_container_width=True)

            st.bar_chart(
                pd.DataFrame({
                    "Score (%)": [
                        accuracy * 100,
                        precision * 100,
                        recall * 100,
                        f1 * 100,
                    ]
                }, index=["Accuracy", "Precision", "Recall", "F1 Score"])
            )

            st.warning(
                "Important: saved model ki training split aur yahan banayi "
                "gayi test split alag ho sakti hain. Is result ko final "
                "scientific evaluation na samjhein. Dataset synthetic hai."
            )

    except Exception as exc:
        st.error(f"Evaluation mein error: {exc}")


import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Rihan | Data Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Python Data Analysis Dashboard")
st.write("Upload a CSV file to explore your data.")

uploaded_file = st.file_uploader(
    "Choose your CSV file",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    st.subheader("Dataset Summary")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isna().sum().sum()))

    st.subheader("Numerical Data Analysis")
    numeric = df.select_dtypes(include="number")

    if not numeric.empty:
        st.bar_chart(numeric)
        st.write(numeric.describe())
    else:
        st.info("No numerical columns found.")

    st.subheader("Download Results")
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "analyzed_data.csv",
        "text/csv"
    )
    