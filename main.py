import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.decomposition import TruncatedSVD


# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="Heart Attack Risk Explorer",
    layout="wide",
)


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


# -----------------------------
# Modeling utilities
# -----------------------------
def build_pipeline(X: pd.DataFrame, cat_cols, num_cols) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return pipe


@st.cache_resource
def train_and_prepare(df: pd.DataFrame):
    # Target
    y = df["target"].astype(int)
    X = df.drop(columns=["target"])

    # Column types (this dataset is classic UCI-style; treat these as categorical)
    # Categorical: sex, cp, fbs, restecg, exang, slope, ca, thal
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Train/test split for model quality display
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline(X, cat_cols, num_cols)
    pipe.fit(X_train, y_train)

    # Evaluate
    proba_test = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba_test)
    pred_test = (proba_test >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred_test)

    # Compute probabilities for all rows (for distribution & percentile)
    proba_all = pipe.predict_proba(X)[:, 1]

    # Permutation importance (on test set)
    # Note: can be a bit slow; keep n_repeats modest
    perm = permutation_importance(
        pipe, X_test, y_test, n_repeats=20, random_state=42, scoring="roc_auc"
    )
    # Importance per original feature (works because we permute raw columns)
    importances = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    # 2D embedding for "proximity" visualization
    # Transform X to model input space then do TruncatedSVD to 2D
    Xt = pipe.named_steps["preprocess"].fit_transform(X)  # fit on full for stable map
    svd = TruncatedSVD(n_components=2, random_state=42)
    emb = svd.fit_transform(Xt)
    emb_df = pd.DataFrame(emb, columns=["dim1", "dim2"])
    emb_df["target"] = y.values
    emb_df["proba"] = proba_all

    # Centroids (in embedding space) for distance-to-risk-group
    pos_centroid = emb_df.loc[emb_df["target"] == 1, ["dim1", "dim2"]].mean().values
    neg_centroid = emb_df.loc[emb_df["target"] == 0, ["dim1", "dim2"]].mean().values

    artifacts = {
        "pipe": pipe,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "auc": auc,
        "acc": acc,
        "proba_all": proba_all,
        "importances": importances,
        "emb_df": emb_df,
        "svd": svd,
        "pos_centroid": pos_centroid,
        "neg_centroid": neg_centroid,
        "X_full": X,
        "y_full": y,
    }
    return artifacts


def make_user_input_form(df: pd.DataFrame, cat_cols, num_cols) -> pd.DataFrame:
    st.sidebar.header("Patient Inputs")

    inputs = {}
    # Numeric inputs: use min/max and median as defaults
    for c in num_cols:
        col = df[c]
        default = float(col.median())
        minv = float(col.min())
        maxv = float(col.max())
        step = (maxv - minv) / 200 if maxv > minv else 1.0
        inputs[c] = st.sidebar.number_input(
            label=c,
            min_value=minv,
            max_value=maxv,
            value=default,
            step=float(step) if step > 0 else 1.0,
        )

    # Categorical inputs: use unique sorted
    for c in cat_cols:
        options = sorted(df[c].unique().tolist())
        default = options[0]
        # Prefer most frequent value as default
        try:
            default = df[c].value_counts().idxmax()
        except Exception:
            pass

        inputs[c] = st.sidebar.selectbox(label=c, options=options, index=options.index(default))

    user_df = pd.DataFrame([inputs])
    return user_df


def plot_probability_gauge(prob: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(prob),
            number={"valueformat": ".3f"},
            title={"text": "Predicted Probability (Heart Attack / Target=1)"},
            gauge={
                "axis": {"range": [0, 1]},
                "steps": [
                    {"range": [0, 0.33]},
                    {"range": [0.33, 0.66]},
                    {"range": [0.66, 1.0]},
                ],
                "threshold": {"line": {"width": 4}, "value": float(prob)},
            },
        )
    )
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def percentile_rank(values: np.ndarray, x: float) -> float:
    return float((values <= x).mean() * 100.0)


# -----------------------------
# Main
# -----------------------------
st.title("Heart Attack Data Analysis Web App (Interactive)")

csv_path = "Heart Attack Data Set.csv"
df = load_data(csv_path)

st.caption(f"Loaded dataset: {csv_path} | shape={df.shape[0]} rows × {df.shape[1]} cols")

# Train model + prepare artifacts
art = train_and_prepare(df)

pipe = art["pipe"]
cat_cols = art["cat_cols"]
num_cols = art["num_cols"]

# Sidebar user input
user_df = make_user_input_form(df.drop(columns=["target"]), cat_cols, num_cols)

# Predict for user
user_prob = float(pipe.predict_proba(user_df)[:, 1][0])

# Layout
colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Model Quality (reference)")
    m1, m2 = st.columns(2)
    m1.metric("ROC-AUC (test split)", f"{art['auc']:.3f}")
    m2.metric("Accuracy (test split)", f"{art['acc']:.3f}")
    st.caption("This is a lightweight reference model (Logistic Regression + scaling + one-hot).")

    st.subheader("Your Predicted Risk Score")
    st.plotly_chart(plot_probability_gauge(user_prob), use_container_width=True)

    # Percentile vs population
    all_probs = art["proba_all"]
    p_all = percentile_rank(all_probs, user_prob)
    p_pos = percentile_rank(all_probs[art["y_full"].values == 1], user_prob)
    p_neg = percentile_rank(all_probs[art["y_full"].values == 0], user_prob)

    st.write(
        f"""
**Percentile position** (higher = closer to high-risk tail):
- Among **all** patients: **{p_all:.1f}th percentile**
- Among **target=1** patients: **{p_pos:.1f}th percentile**
- Among **target=0** patients: **{p_neg:.1f}th percentile**
"""
    )

with colB:
    st.subheader("Risk Score Distribution (All Patients)")
    dist_df = pd.DataFrame({"probability": all_probs, "target": art["y_full"].values.astype(int)})
    fig_hist = px.histogram(
        dist_df,
        x="probability",
        color="target",
        nbins=30,
        barmode="overlay",
        opacity=0.6,
        hover_data=["probability"],
    )
    fig_hist.add_vline(x=user_prob, line_width=3)
    fig_hist.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

# Feature importance section
st.divider()
st.subheader("Which Features Influence the Outcome Most? (Permutation Importance)")

imp = art["importances"].copy()
imp = imp[imp["importance_mean"] > 0].head(12)

fig_imp = px.bar(
    imp,
    x="importance_mean",
    y="feature",
    orientation="h",
    error_x="importance_std",
    title="Top features by mean permutation importance (ROC-AUC drop)",
)
fig_imp.update_layout(height=450, margin=dict(l=20, r=20, t=60, b=20))
st.plotly_chart(fig_imp, use_container_width=True)

st.caption(
    "Permutation importance measures how much model performance drops when each feature is shuffled. "
    "Higher values imply stronger influence under this model."
)

# Proximity visualization (2D embedding)
st.divider()
st.subheader("How Close Are You to the High-Risk Group? (2D Proximity Map)")

# Compute user's embedding point
Xt_user = pipe.named_steps["preprocess"].transform(user_df)
user_emb = art["svd"].transform(Xt_user)[0]

emb_df = art["emb_df"].copy()
emb_df["group"] = emb_df["target"].map({0: "target=0", 1: "target=1"})

# Distances to centroids in 2D
pos_centroid = art["pos_centroid"]
neg_centroid = art["neg_centroid"]
d_pos = float(np.linalg.norm(user_emb - pos_centroid))
d_neg = float(np.linalg.norm(user_emb - neg_centroid))

# Turn into a simple "closeness" score (0~100): closer to positive centroid => higher
closeness = 100.0 * (d_neg / (d_pos + d_neg + 1e-9))
closeness = float(np.clip(closeness, 0, 100))

m3, m4, m5 = st.columns(3)
m3.metric("Distance to target=1 centroid", f"{d_pos:.3f}")
m4.metric("Distance to target=0 centroid", f"{d_neg:.3f}")
m5.metric("Closeness to target=1 group (0-100)", f"{closeness:.1f}")

fig_scatter = px.scatter(
    emb_df,
    x="dim1",
    y="dim2",
    color="group",
    hover_data={"proba": ":.3f"},
    title="Patients in a 2D projection of model input space (TruncatedSVD)",
    opacity=0.65,
)
# Add centroids
fig_scatter.add_trace(
    go.Scatter(
        x=[pos_centroid[0]],
        y=[pos_centroid[1]],
        mode="markers",
        marker=dict(size=14, symbol="x"),
        name="Centroid: target=1",
        hoverinfo="skip",
    )
)
fig_scatter.add_trace(
    go.Scatter(
        x=[neg_centroid[0]],
        y=[neg_centroid[1]],
        mode="markers",
        marker=dict(size=14, symbol="x"),
        name="Centroid: target=0",
        hoverinfo="skip",
    )
)
# Add user point
fig_scatter.add_trace(
    go.Scatter(
        x=[user_emb[0]],
        y=[user_emb[1]],
        mode="markers",
        marker=dict(size=16, symbol="star"),
        name="You",
        hovertemplate="You<br>dim1=%{x:.3f}<br>dim2=%{y:.3f}<extra></extra>",
    )
)
fig_scatter.update_layout(height=560, margin=dict(l=20, r=20, t=60, b=20))
st.plotly_chart(fig_scatter, use_container_width=True)

st.caption(
    "This proximity map is a visualization aid: it projects the model’s processed feature space to 2D. "
    "Closeness is computed relative to class centroids in this 2D space for interpretability."
)

# Data preview (optional)
with st.expander("Data Preview"):
    st.dataframe(df.head(20), use_container_width=True)
