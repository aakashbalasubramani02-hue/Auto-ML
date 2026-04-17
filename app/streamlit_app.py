import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Anchor cwd to repo root so models/ and data/ resolve correctly
# on both local machines and Hugging Face Spaces.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

try:
    from src.meta_feature_extractor import extract_meta_features
    from src.dataset_analyzer import analyze_dataset, compute_complexity_score
    from src.predictor import predict_algorithms, load_meta_model
    from src.performance_predictor import predict_performance
except Exception as e:
    st.error(f"❌ Failed to load backend modules: {e}")
    st.stop()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AutoML Algorithm Recommender", layout="wide")
st.title("🤖 AutoML Algorithm Recommendation System")
st.markdown("### Using Meta-Learning to Predict the Best ML Algorithms")

# ── Sidebar upload ────────────────────────────────────────────────────────────
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# ── Landing page ──────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Please upload a CSV dataset from the sidebar to get started.")
    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. **Upload** your CSV dataset using the sidebar
    2. **Select** the target column (the variable you want to predict)
    3. **Click** 'Analyze Dataset & Recommend Algorithms'
    4. **View** the top 3 recommended algorithms with expected performance
    """)
    st.markdown("### 🔧 Features")
    st.markdown("""
    - **Meta-Learning**: Uses XGBoost to learn from dataset characteristics
    - **Smart Recommendations**: Predicts the best algorithms for your data
    - **Performance Estimation**: Estimates expected accuracy for each algorithm
    - **Interactive Visualizations**: Explore your data with charts and heatmaps
    - **Complexity Analysis**: Understand how difficult your ML task is
    """)
    st.stop()

# ── Load CSV ──────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"❌ Could not read the file: {e}")
    st.stop()

if df.empty:
    st.error("❌ The uploaded file is empty. Please upload a valid dataset.")
    st.stop()

if len(df.columns) < 2:
    st.error("❌ Dataset must have at least 2 columns (features + target).")
    st.stop()

st.success(f"✅ Dataset loaded — {df.shape[0]} rows × {df.shape[1]} columns")

# ── Dataset preview ───────────────────────────────────────────────────────────
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# ── Missing value warning ─────────────────────────────────────────────────────
total_missing = int(df.isnull().sum().sum())
if total_missing > 0:
    st.warning(f"⚠️ Dataset contains {total_missing} missing value(s). "
               "Numeric columns will be filled with column median before analysis.")
    numeric_cols_all = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols_all] = df[numeric_cols_all].fillna(df[numeric_cols_all].median())

# ── Numeric column check ──────────────────────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    st.error("❌ Dataset must contain at least one numeric feature column.")
    st.stop()

# ── Target column selection ───────────────────────────────────────────────────
st.subheader("🎯 Select Target Column")
target_column = st.selectbox("Choose the target variable:", ["-- select --"] + df.columns.tolist())

if target_column == "-- select --":
    st.warning("⚠️ Please select a target column to continue.")
    st.stop()

if target_column not in df.columns:
    st.error(f"❌ Column '{target_column}' not found in dataset.")
    st.stop()

# ── Analyze button ────────────────────────────────────────────────────────────
if not st.button("🔍 Analyze Dataset & Recommend Algorithms"):
    st.stop()

# ── Everything below runs only after button click ─────────────────────────────
with st.spinner("Analyzing dataset..."):

    # ── Meta-feature extraction ───────────────────────────────────────────────
    try:
        meta_features = extract_meta_features(df, target_column)
    except Exception as e:
        st.error(f"❌ Meta-feature extraction failed: {e}")
        st.stop()

    # ── Dataset analysis ──────────────────────────────────────────────────────
    try:
        analysis  = analyze_dataset(df, target_column)
        complexity = compute_complexity_score(meta_features)
    except Exception as e:
        st.error(f"❌ Dataset analysis failed: {e}")
        st.stop()

st.success("✅ Analysis complete!")

# ── Summary + meta-features ───────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Dataset Summary")
    st.write(f"**Rows:** {analysis['shape'][0]}")
    st.write(f"**Columns:** {analysis['shape'][1]}")
    st.write(f"**Missing Values:** {analysis['total_missing']}")
    st.write(f"**Number of Classes:** {analysis.get('n_classes', 'N/A')}")
    st.write(f"**Complexity:** {complexity}")

with col2:
    st.subheader("🔬 Extracted Meta-Features")
    meta_display = {col: round(float(meta_features[col].values[0]), 4)
                    for col in meta_features.columns}
    st.write(meta_display)

# ── Visualizations ────────────────────────────────────────────────────────────
st.subheader("📊 Visualizations")
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    if analysis.get("class_distribution"):
        try:
            st.write("**Class Distribution**")
            fig, ax = plt.subplots(figsize=(8, 4))
            pd.Series(analysis["class_distribution"]).plot(
                kind="bar", ax=ax, color="steelblue"
            )
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            ax.set_title("Target Class Distribution")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"⚠️ Could not render class distribution chart: {e}")

with viz_col2:
    corr_matrix = analysis.get("correlation_matrix")
    numeric_feature_cols = [c for c in numeric_cols if c != target_column]

    if corr_matrix is not None and len(numeric_feature_cols) >= 2:
        try:
            st.write("**Feature Correlation Heatmap**")
            if corr_matrix.shape[0] > 10:
                corr_matrix = corr_matrix.iloc[:10, :10]
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                corr_matrix, annot=True, fmt=".2f",
                cmap="coolwarm", ax=ax, cbar_kws={"shrink": 0.8}
            )
            ax.set_title("Feature Correlations")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"⚠️ Could not render correlation heatmap: {e}")
    else:
        st.info("ℹ️ Not enough numeric features to generate a correlation heatmap.")

st.markdown("---")

# ── Algorithm recommendations ─────────────────────────────────────────────────
st.subheader("🎯 Top 3 Recommended Algorithms")

try:
    recommendations = predict_algorithms(meta_features, top_k=3)
except FileNotFoundError:
    st.error("❌ Model files not found. Please train the model first: `python src/train_model.py`")
    st.stop()
except Exception as e:
    st.error(f"❌ Prediction failed: {e}")
    st.stop()

if not recommendations:
    st.error("❌ No recommendations returned. Please check the model.")
    st.stop()

for i, rec in enumerate(recommendations, 1):
    algo_name  = rec.get("algorithm", "Unknown")
    confidence = float(rec.get("confidence", 0))

    try:
        expected_acc = float(predict_performance(meta_features, algo_name))
    except Exception:
        expected_acc = 0.0

    st.markdown(f"### {i}. {algo_name}")
    if i == 1:
        st.success(f"⭐ Recommended Algorithm: **{algo_name}**")

    rc1, rc2 = st.columns(2)
    with rc1:
        st.metric("Confidence", f"{confidence:.2%}")
    with rc2:
        st.metric("Expected Accuracy", f"{expected_acc:.2%}")

    st.progress(min(max(confidence, 0.0), 1.0))
    st.markdown("---")

# ── Feature importance ────────────────────────────────────────────────────────
st.subheader("📊 Meta-Model Feature Importance")

try:
    model, _ = load_meta_model()
    classifier = model.named_steps["classifier"] if hasattr(model, "named_steps") else model

    if not hasattr(classifier, "feature_importances_"):
        st.info("ℹ️ Feature importance is not available for this model type.")
    else:
        feature_names = ["n_samples", "n_features", "imbalance_ratio", "avg_corr", "class_sep"]
        importances   = classifier.feature_importances_
        indices       = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(
            range(len(importances)),
            importances[indices],
            color="teal", edgecolor="white"
        )
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
        ax.set_xlabel("Meta-Features")
        ax.set_ylabel("Importance")
        ax.set_title("Feature Importance in Meta-Model")

        for bar, val in zip(bars, importances[indices]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

except Exception as e:
    st.warning(f"⚠️ Feature importance could not be displayed: {e}")
