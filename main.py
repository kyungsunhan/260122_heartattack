import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.decomposition import TruncatedSVD


# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="심근경색 위험도 탐색기",
    layout="wide",
)


# -----------------------------
# 심장 일러스트 (내장 SVG)
# -----------------------------
HEART_SVG = """
<svg width="520" height="140" viewBox="0 0 520 140" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g" x1="0" x2="1">
      <stop offset="0" stop-color="#ff5a5f"/>
      <stop offset="1" stop-color="#d7263d"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="520" height="140" rx="18" fill="#f7f7f9"/>
  <path d="M150 55
           C150 30, 180 20, 200 35
           C220 15, 255 25, 255 55
           C255 85, 220 100, 200 118
           C180 100, 150 85, 150 55 Z"
        fill="url(#g)" opacity="0.95"/>
  <path d="M80 75 H118 L130 52 L145 98 L160 70 H210"
        fill="none" stroke="#111827" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>
  <text x="260" y="58" font-family="Arial" font-size="22" fill="#111827" font-weight="700">
    심근경색 위험도 탐색기
  </text>
  <text x="260" y="86" font-family="Arial" font-size="14" fill="#374151">
    데이터 기반 예측 점수 · 변수 영향 · 개인별 설명(SHAP)
  </text>
</svg>
"""


# -----------------------------
# 한글 변수명 매핑
# -----------------------------
# (본 데이터셋은 UCI heart 계열 컬럼을 가정합니다)
KOR_COL = {
    "age": "나이",
    "sex": "성별",
    "cp": "흉통 유형(cp)",
    "trestbps": "안정시 혈압(trestbps)",
    "chol": "혈청 콜레스테롤(chol)",
    "fbs": "공복혈당>120 여부(fbs)",
    "restecg": "휴식 심전도(restecg)",
    "thalach": "최대 심박수(thalach)",
    "exang": "운동유발 협심증(exang)",
    "oldpeak": "운동 후 ST 저하(oldpeak)",
    "slope": "ST 기울기(slope)",
    "ca": "주요 혈관수(ca)",
    "thal": "Thal 검사(thal)",
    "target": "심근경색/심장질환(타깃)",
}

# 범주형 컬럼 설명(옵션)
KOR_CAT_DESC = {
    "sex": "0=여성, 1=남성(데이터 정의에 따름)",
    "fbs": "1=공복혈당>120mg/dL, 0=그 외",
    "exang": "1=있음, 0=없음",
}


# -----------------------------
# 데이터 로딩
# -----------------------------
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


# -----------------------------
# 모델 파이프라인
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


def percentile_rank(values: np.ndarray, x: float) -> float:
    return float((values <= x).mean() * 100.0)


def plot_probability_gauge(prob: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(prob),
            number={"valueformat": ".3f"},
            title={"text": "예측 위험도(확률) — 타깃=1"},
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


def _pretty_kor_feature_name(raw_name: str) -> str:
    """
    preprocess.get_feature_names_out() 결과를 한글로 보기 좋게 변환
    예: 'num__age' -> '나이'
        'cat__sex_1' -> '성별=1'
    """
    # 기본 형태: "num__age", "cat__sex_1"
    if raw_name.startswith("num__"):
        col = raw_name.replace("num__", "")
        return KOR_COL.get(col, col)

    if raw_name.startswith("cat__"):
        rest = raw_name.replace("cat__", "")
        # rest may be like sex_1, cp_2, thal_3 ...
        parts = rest.split("_", 1)
        col = parts[0]
        val = parts[1] if len(parts) > 1 else ""
        base = KOR_COL.get(col, col)
        return f"{base}={val}" if val != "" else base

    # fallback
    return KOR_COL.get(raw_name, raw_name)


@st.cache_resource
def train_and_prepare(df: pd.DataFrame):
    y = df["target"].astype(int)
    X = df.drop(columns=["target"])

    # 범주형/연속형 구분 (UCI 스타일)
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline(X, cat_cols, num_cols)
    pipe.fit(X_train, y_train)

    proba_test = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba_test)
    pred_test = (proba_test >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred_test)

    # 전체 예측확률
    proba_all = pipe.predict_proba(X)[:, 1]

    # Permutation importance (원본 컬럼 기준)
    perm = permutation_importance(
        pipe, X_test, y_test, n_repeats=20, random_state=42, scoring="roc_auc"
    )
    importances = pd.DataFrame(
        {
            "feature": X.columns,
            "feature_kor": [KOR_COL.get(c, c) for c in X.columns],
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    # 2D 임베딩
    Xt = pipe.named_steps["preprocess"].fit_transform(X)
    svd = TruncatedSVD(n_components=2, random_state=42)
    emb = svd.fit_transform(Xt)
    emb_df = pd.DataFrame(emb, columns=["dim1", "dim2"])
    emb_df["target"] = y.values
    emb_df["proba"] = proba_all

    pos_centroid = emb_df.loc[emb_df["target"] == 1, ["dim1", "dim2"]].mean().values
    neg_centroid = emb_df.loc[emb_df["target"] == 0, ["dim1", "dim2"]].mean().values

    # SHAP (선형모델용)
    explainer = shap.LinearExplainer(
        pipe.named_steps["model"],
        pipe.named_steps["preprocess"].transform(X),
        feature_perturbation="interventional",
    )

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
        "explainer": explainer,
    }
    return artifacts


def make_user_input_form(X: pd.DataFrame, cat_cols, num_cols) -> pd.DataFrame:
    st.sidebar.header("환자 특성 입력")

    inputs = {}

    st.sidebar.caption("연속형 변수")
    for c in num_cols:
        col = X[c]
        label = KOR_COL.get(c, c)
        default = float(col.median())
        minv = float(col.min())
        maxv = float(col.max())
        step = (maxv - minv) / 200 if maxv > minv else 1.0

        inputs[c] = st.sidebar.number_input(
            label=f"{label}",
            min_value=minv,
            max_value=maxv,
            value=default,
            step=float(step) if step > 0 else 1.0,
        )

    st.sidebar.divider()
    st.sidebar.caption("범주형 변수")
    for c in cat_cols:
        label = KOR_COL.get(c, c)
        options = sorted(X[c].unique().tolist())

        # 최빈값 기본 선택
        try:
            default = X[c].value_counts().idxmax()
        except Exception:
            default = options[0]

        # 설명(있으면)
        if c in KOR_CAT_DESC:
            st.sidebar.caption(f"{label}: {KOR_CAT_DESC[c]}")

        inputs[c] = st.sidebar.selectbox(
            label=f"{label}",
            options=options,
            index=options.index(default) if default in options else 0,
        )

    user_df = pd.DataFrame([inputs])
    return user_df


# -----------------------------
# 메인 UI
# -----------------------------
st.markdown(HEART_SVG, unsafe_allow_html=True)

csv_path = "Heart Attack Data Set.csv"
df = load_data(csv_path)

st.caption(f"불러온 데이터: {csv_path} | {df.shape[0]}행 × {df.shape[1]}열")

# 모델 학습 + 아티팩트 준비
art = train_and_prepare(df)
pipe = art["pipe"]
cat_cols = art["cat_cols"]
num_cols = art["num_cols"]

# 사용자 입력
user_df = make_user_input_form(df.drop(columns=["target"]), cat_cols, num_cols)

# 예측
user_prob = float(pipe.predict_proba(user_df)[:, 1][0])

# 상단 2열 구성
colA, colB = st.columns([1, 1])

with colA:
    st.subheader("모델 성능(참고용)")
    m1, m2 = st.columns(2)
    m1.metric("ROC-AUC(테스트 분할)", f"{art['auc']:.3f}")
    m2.metric("정확도(테스트 분할)", f"{art['acc']:.3f}")
    st.caption("간단한 기준 모델(로지스틱 회귀 + 표준화 + 원-핫 인코딩)입니다.")

    st.subheader("나의 예측 위험도 점수")
    st.plotly_chart(plot_probability_gauge(user_prob), use_container_width=True)

    all_probs = art["proba_all"]
    p_all = percentile_rank(all_probs, user_prob)
    p_pos = percentile_rank(all_probs[art["y_full"].values == 1], user_prob)
    p_neg = percentile_rank(all_probs[art["y_full"].values == 0], user_prob)

    st.write(
        f"""
**퍼센타일 위치(높을수록 고위험 쪽 꼬리에 가까움)**  
- 전체 환자 기준: **{p_all:.1f} 퍼센타일**  
- 타깃=1(발생군) 기준: **{p_pos:.1f} 퍼센타일**  
- 타깃=0(비발생군) 기준: **{p_neg:.1f} 퍼센타일**
"""
    )

with colB:
    st.subheader("전체 분포에서 나의 위치(위험도 분포)")
    dist_df = pd.DataFrame(
        {"예측확률": all_probs, "타깃": art["y_full"].values.astype(int)}
    )
    dist_df["군"] = dist_df["타깃"].map({0: "비발생군(0)", 1: "발생군(1)"})

    fig_hist = px.histogram(
        dist_df,
        x="예측확률",
        color="군",
        nbins=30,
        barmode="overlay",
        opacity=0.6,
        hover_data=["예측확률", "군"],
    )
    fig_hist.add_vline(x=user_prob, line_width=3)
    fig_hist.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------
# 전역 변수 중요도
# -----------------------------
st.divider()
st.subheader("어떤 변수가 ‘발생(타깃=1)’에 크게 영향을 주는가? (Permutation Importance)")

imp = art["importances"].copy()
imp = imp[imp["importance_mean"] > 0].head(12)

fig_imp = px.bar(
    imp,
    x="importance_mean",
    y="feature_kor",
    orientation="h",
    error_x="importance_std",
    title="상위 변수 중요도(변수를 섞었을 때 ROC-AUC 감소량 기준)",
    labels={"importance_mean": "중요도(평균)", "feature_kor": "변수"},
)
fig_imp.update_layout(height=450, margin=dict(l=20, r=20, t=60, b=20))
st.plotly_chart(fig_imp, use_container_width=True)

st.caption(
    "Permutation importance는 각 변수를 무작위로 섞었을 때 모델 성능이 얼마나 감소하는지를 측정합니다. 값이 클수록 해당 변수가 예측에 더 중요합니다."
)

# -----------------------------
# 군집 근접도(직관적 위치)
# -----------------------------
st.divider()
st.subheader("나는 고위험군(발생군)과 얼마나 가까운가? (2차원 근접도 지도)")

Xt_user = pipe.named_steps["preprocess"].transform(user_df)
user_emb = art["svd"].transform(Xt_user)[0]

emb_df = art["emb_df"].copy()
emb_df["군"] = emb_df["target"].map({0: "비발생군(0)", 1: "발생군(1)"})

pos_centroid = art["pos_centroid"]
neg_centroid = art["neg_centroid"]
d_pos = float(np.linalg.norm(user_emb - pos_centroid))
d_neg = float(np.linalg.norm(user_emb - neg_centroid))

# 가까움 점수(0~100): 발생군 중심에 가까울수록 점수↑
closeness = 100.0 * (d_neg / (d_pos + d_neg + 1e-9))
closeness = float(np.clip(closeness, 0, 100))

m3, m4, m5 = st.columns(3)
m3.metric("발생군 중심까지 거리", f"{d_pos:.3f}")
m4.metric("비발생군 중심까지 거리", f"{d_neg:.3f}")
m5.metric("발생군 근접도(0-100)", f"{closeness:.1f}")

fig_scatter = px.scatter(
    emb_df,
    x="dim1",
    y="dim2",
    color="군",
    hover_data={"proba": ":.3f"},
    title="모델 입력공간을 2차원으로 축소한 환자 분포(시각화 목적)",
    opacity=0.65,
    labels={"dim1": "차원 1", "dim2": "차원 2"},
)
fig_scatter.add_trace(
    go.Scatter(
        x=[pos_centroid[0]],
        y=[pos_centroid[1]],
        mode="markers",
        marker=dict(size=14, symbol="x"),
        name="발생군 중심",
        hoverinfo="skip",
    )
)
fig_scatter.add_trace(
    go.Scatter(
        x=[neg_centroid[0]],
        y=[neg_centroid[1]],
        mode="markers",
        marker=dict(size=14, symbol="x"),
        name="비발생군 중심",
        hoverinfo="skip",
    )
)
fig_scatter.add_trace(
    go.Scatter(
        x=[user_emb[0]],
        y=[user_emb[1]],
        mode="markers",
        marker=dict(size=16, symbol="star"),
        name="나(입력값)",
        hovertemplate="나(입력값)<br>차원1=%{x:.3f}<br>차원2=%{y:.3f}<extra></extra>",
    )
)
fig_scatter.update_layout(height=560, margin=dict(l=20, r=20, t=60, b=20))
st.plotly_chart(fig_scatter, use_container_width=True)

st.caption(
    "이 지도는 해석 보조용 시각화입니다. 모델 입력공간(전처리된 특성)을 2차원으로 축소하여 '발생군/비발생군' 분포와 나의 위치를 직관적으로 보여줍니다."
)

# -----------------------------
# SHAP 개인별 설명
# -----------------------------
st.divider()
st.subheader("왜 이런 점수가 나왔나? (개인별 SHAP 설명)")

explainer = art["explainer"]
shap_values_user = explainer.shap_values(Xt_user)

# 전처리 후 feature name을 한글로 보기 좋게 변환
raw_feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
feature_names_kor = [_pretty_kor_feature_name(n) for n in raw_feature_names]

fig, ax = plt.subplots(figsize=(10, 5))
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_user[0],
        base_values=explainer.expected_value,
        data=Xt_user[0],
        feature_names=feature_names_kor,
    ),
    max_display=12,
    show=False,
)
st.pyplot(fig)
plt.close(fig)

st.caption(
    "빨간색은 예측 위험도를 올리는 방향, 파란색은 내리는 방향입니다. "
    "기여도는 모델의 로짓(log-odds) 공간에서 계산됩니다."
)

# -----------------------------
# SHAP 전역 영향(보완)
# -----------------------------
st.subheader("전반적으로 중요한 변수는? (SHAP 전역 영향)")
X_sample = art["X_full"].sample(n=min(200, len(art["X_full"])), random_state=42)
X_sample_proc = pipe.named_steps["preprocess"].transform(X_sample)
shap_values_sample = explainer.shap_values(X_sample_proc)

fig2, ax2 = plt.subplots(figsize=(10, 5))
shap.plots.bar(
    shap.Explanation(
        values=shap_values_sample,
        base_values=explainer.expected_value,
        data=X_sample_proc,
        feature_names=feature_names_kor,
    ),
    max_display=12,
    show=False,
)
st.pyplot(fig2)
plt.close(fig2)

st.caption("SHAP 전역 영향은 변수의 평균적 기여 크기를 요약합니다(방향성/기여도 해석에 유용).")

# -----------------------------
# 데이터 미리보기
# -----------------------------
with st.expander("데이터 미리보기(상위 20행)"):
    # 컬럼명 표시를 한글로 바꿔서 보여주되, 실제 연산은 영문 컬럼명 사용
    preview = df.copy()
    preview.columns = [KOR_COL.get(c, c) for c in preview.columns]
    st.dataframe(preview.head(20), use_container_width=True)
