import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import shap

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
# 한글 변수명 매핑
# -----------------------------
KOR_COL = {
    "age": "나이",
    "sex": "성별",
    "cp": "흉통 유형",
    "trestbps": "안정시 혈압",
    "chol": "혈청 콜레스테롤",
    "fbs": "공복혈당 >120 여부",
    "restecg": "휴식 심전도",
    "thalach": "최대 심박수",
    "exang": "운동유발 협심증",
    "oldpeak": "운동 후 ST 저하",
    "slope": "ST 기울기",
    "ca": "조영된 주요 혈관 수",
    "thal": "Thal 검사",
    "target": "타깃(발생=1)",
}

# -----------------------------
# 범주형 코드 -> 한글 의미(표준 UCI 정의 기반; 데이터셋에 따라 일부 수정 가능)
# -----------------------------
CAT_VALUE_LABELS = {
    "sex": {0: "여성(0)", 1: "남성(1)"},
    "cp": {
        0: "전형적 협심증(0)",
        1: "비전형적 협심증(1)",
        2: "비협심증성 흉통(2)",
        3: "무증상(3)",
    },
    "fbs": {0: "아니오(0): ≤120 mg/dL", 1: "예(1): >120 mg/dL"},
    "restecg": {0: "정상(0)", 1: "ST-T 이상(1)", 2: "좌심실비대 가능(2)"},
    "exang": {0: "없음(0)", 1: "있음(1)"},
    "slope": {0: "상승형(0)", 1: "평탄형(1)", 2: "하강형(2)"},
    "ca": {0: "0개(0)", 1: "1개(1)", 2: "2개(2)", 3: "3개(3)", 4: "미상/코드값(4)"},
    "thal": {0: "미상(0)", 1: "정상(1)", 2: "고정 결손(2)", 3: "가역 결손(3)"},
}

# -----------------------------
# 일반인이 입력/해석하기 어려운 변수 목록(경고 강조용)
# -----------------------------
HARD_VARS = ["oldpeak", "slope", "ca", "thal", "restecg"]
HARD_VARS_KOR = ", ".join([KOR_COL[v] for v in HARD_VARS])


# -----------------------------
# 데이터 로딩
# -----------------------------
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


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
    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


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
    preprocess.get_feature_names_out() 결과를 한글로 변환
    예: 'num__age' -> '나이'
        'cat__sex_1' -> '성별=남성(1)'
    """
    if raw_name.startswith("num__"):
        col = raw_name.replace("num__", "")
        return KOR_COL.get(col, col)

    if raw_name.startswith("cat__"):
        rest = raw_name.replace("cat__", "")
        parts = rest.split("_", 1)
        col = parts[0]
        val = parts[1] if len(parts) > 1 else ""
        base = KOR_COL.get(col, col)

        if val == "":
            return base

        try:
            v = int(val)
        except Exception:
            v = val

        if isinstance(v, int) and col in CAT_VALUE_LABELS:
            vlabel = CAT_VALUE_LABELS[col].get(v, f"{v}")
            return f"{base}={vlabel}"

        return f"{base}={val}"

    return raw_name


@st.cache_resource
def train_and_prepare(df: pd.DataFrame):
    y = df["target"].astype(int)
    X = df.drop(columns=["target"])

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

    proba_all = pipe.predict_proba(X)[:, 1]

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

    Xt_full = pipe.named_steps["preprocess"].fit_transform(X)
    svd = TruncatedSVD(n_components=2, random_state=42)
    emb = svd.fit_transform(Xt_full)
    emb_df = pd.DataFrame(emb, columns=["dim1", "dim2"])
    emb_df["target"] = y.values
    emb_df["proba"] = proba_all

    pos_centroid = emb_df.loc[emb_df["target"] == 1, ["dim1", "dim2"]].mean().values
    neg_centroid = emb_df.loc[emb_df["target"] == 0, ["dim1", "dim2"]].mean().values

    explainer = shap.LinearExplainer(
        pipe.named_steps["model"],
        pipe.named_steps["preprocess"].transform(X),
        feature_perturbation="interventional",
    )

    return {
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


def make_user_input_form(X: pd.DataFrame, cat_cols, num_cols) -> pd.DataFrame:
    st.sidebar.header("환자 특성 입력")

    # ---- 강력 경고(강조) ----
    st.sidebar.warning(
        f"다음 항목은 일반인이 정확히 해석/입력하기 어려울 수 있습니다: **{HARD_VARS_KOR}**\n\n"
        "해당 값은 검사 결과/의무기록 기반으로 입력하는 것이 바람직하며, "
        "**불확실한 경우 반드시 주치의와 상의** 후 입력하세요."
    )

    inputs = {}

    st.sidebar.caption("연속형 변수")
    for c in num_cols:
        col = X[c]
        label = KOR_COL.get(c, c)
        default = float(col.median())
        minv = float(col.min())
        maxv = float(col.max())
        step = (maxv - minv) / 200 if maxv > minv else 1.0

        # ST 저하(oldpeak)는 특히 어려운 항목이므로 추가 캡션
        if c == "oldpeak":
            st.sidebar.caption("※ '운동 후 ST 저하'는 운동부하검사/심전도 판독에 기반합니다. 불확실하면 주치의와 상의하세요.")

        inputs[c] = st.sidebar.number_input(
            label=label,
            min_value=float(minv),
            max_value=float(maxv),
            value=float(default),
            step=float(step) if step > 0 else 1.0,
        )

    st.sidebar.divider()
    st.sidebar.caption("범주형 변수(코드 → 의미)")

    # 범주형 중에서도 이해 난이도 높은 항목 안내
    st.sidebar.info(
        "참고: **휴식 심전도, ST 기울기, 조영된 주요 혈관 수, Thal 검사**는 "
        "검사/영상 판독 결과에 기반하는 경우가 많습니다. 모르는 값은 주치의와 상의하세요."
    )

    for c in cat_cols:
        label = KOR_COL.get(c, c)
        observed_codes = sorted(pd.unique(X[c]).tolist())
        mapper = CAT_VALUE_LABELS.get(c, {})

        display_options = []
        code_by_display = {}
        for code in observed_codes:
            try:
                code_int = int(code)
            except Exception:
                code_int = code

            display = mapper.get(code_int, f"코드 {code_int}")
            display_options.append(display)
            code_by_display[display] = code_int

        # 최빈값
        try:
            default_code = int(X[c].value_counts().idxmax())
        except Exception:
            default_code = observed_codes[0]
            try:
                default_code = int(default_code)
            except Exception:
                pass

        default_display = mapper.get(default_code, f"코드 {default_code}")
        if default_display not in display_options:
            default_display = display_options[0]

        chosen_display = st.sidebar.selectbox(
            label=label,
            options=display_options,
            index=display_options.index(default_display),
        )

        inputs[c] = code_by_display[chosen_display]

    return pd.DataFrame([inputs])


# -----------------------------
# 메인 헤더: 작은 아이콘 + 제목
# -----------------------------
header_col1, header_col2 = st.columns([1, 10])

with header_col1:
    st.image("heart.png", width=80)  # 아이콘 크기 조절 (60~100 권장)

with header_col2:
    st.markdown(
        """
        <h1 style="margin-bottom: 0;">심근경색 위험도 탐색기</h1>
        """,
        unsafe_allow_html=True,
    )


CSV_PATH = "Heart Attack Data Set.csv"
df = load_data(CSV_PATH)
st.caption(f"불러온 데이터: {CSV_PATH} | {df.shape[0]}행 × {df.shape[1]}열")

art = train_and_prepare(df)
pipe = art["pipe"]

# 사용자 입력
user_df = make_user_input_form(df.drop(columns=["target"]), art["cat_cols"], art["num_cols"])
user_prob = float(pipe.predict_proba(user_df)[:, 1][0])

# 상단: 성능 + 예측/분포
colA, colB = st.columns([1, 1])

with colA:
    st.subheader("모델 성능(참고)")
    m1, m2 = st.columns(2)
    m1.metric("ROC-AUC(테스트 분할)", f"{art['auc']:.3f}")
    m2.metric("정확도(테스트 분할)", f"{art['acc']:.3f}")

    st.subheader("나의 예측 위험도 점수")
    st.plotly_chart(plot_probability_gauge(user_prob), use_container_width=True)

    all_probs = art["proba_all"]
    p_all = percentile_rank(all_probs, user_prob)
    p_pos = percentile_rank(all_probs[art["y_full"].values == 1], user_prob)
    p_neg = percentile_rank(all_probs[art["y_full"].values == 0], user_prob)

    st.write(
        f"""
**퍼센타일 위치(높을수록 고위험 쪽에 가까움)**  
- 전체 환자 기준: **{p_all:.1f} 퍼센타일**  
- 발생군(타깃=1) 기준: **{p_pos:.1f} 퍼센타일**  
- 비발생군(타깃=0) 기준: **{p_neg:.1f} 퍼센타일**
"""
    )

with colB:
    st.subheader("전체 분포에서 나의 위치")
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
# 전역 변수 중요도 (Permutation)
# -----------------------------
st.divider()
st.subheader("변수 중요도(전역) — Permutation Importance")

imp = art["importances"].copy()
imp = imp[imp["importance_mean"] > 0].head(12)

fig_imp = px.bar(
    imp,
    x="importance_mean",
    y="feature_kor",
    orientation="h",
    error_x="importance_std",
    labels={"importance_mean": "중요도(평균)", "feature_kor": "변수"},
    title="상위 중요 변수(변수를 섞었을 때 ROC-AUC 감소량 기준)",
)
fig_imp.update_layout(height=450, margin=dict(l=20, r=20, t=60, b=20))
st.plotly_chart(fig_imp, use_container_width=True)


# -----------------------------
# 근접도 지도(2D)
# -----------------------------
st.divider()
st.subheader("고위험군(발생군)과의 근접도")

Xt_user = pipe.named_steps["preprocess"].transform(user_df)
user_emb = art["svd"].transform(Xt_user)[0]

emb_df = art["emb_df"].copy()
emb_df["군"] = emb_df["target"].map({0: "비발생군(0)", 1: "발생군(1)"})

pos_centroid = art["pos_centroid"]
neg_centroid = art["neg_centroid"]
d_pos = float(np.linalg.norm(user_emb - pos_centroid))
d_neg = float(np.linalg.norm(user_emb - neg_centroid))

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
    labels={"dim1": "차원 1", "dim2": "차원 2"},
    title="환자 분포(2D 투영) 및 나의 위치",
    opacity=0.65,
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


# -----------------------------
# SHAP 개인별 설명 (Plotly)
# -----------------------------
st.divider()
st.subheader("개인별 변수 기여도(SHAP)")

explainer = art["explainer"]
shap_values_user = explainer.shap_values(Xt_user)[0]

raw_feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
feature_names_kor = [_pretty_kor_feature_name(n) for n in raw_feature_names]

shap_user_df = pd.DataFrame({"변수": feature_names_kor, "기여도": shap_values_user})
shap_user_df["절대기여도"] = shap_user_df["기여도"].abs()

top_user = (
    shap_user_df.sort_values("절대기여도", ascending=False)
    .head(12)
    .sort_values("기여도")
)
top_user["방향"] = np.where(top_user["기여도"] > 0, "위험 증가", "위험 감소")

fig_shap_user = px.bar(
    top_user,
    x="기여도",
    y="변수",
    orientation="h",
    color="방향",
    title="나의 예측 위험도에 대한 변수 기여도(SHAP)",
    hover_data={"기여도": ":.3f", "방향": True},
)
fig_shap_user.add_vline(x=0, line_width=2, line_dash="dash")
fig_shap_user.update_layout(
    height=520,
    xaxis_title="기여도 (log-odds 기준)",
    yaxis_title="",
    margin=dict(l=20, r=20, t=60, b=20),
)
st.plotly_chart(fig_shap_user, use_container_width=True)


# -----------------------------
# SHAP 전역 중요도 (Plotly)
# -----------------------------
st.subheader("전역 변수 영향(SHAP)")

X_sample = art["X_full"].sample(n=min(200, len(art["X_full"])), random_state=42)
X_sample_proc = pipe.named_steps["preprocess"].transform(X_sample)
shap_values_sample = explainer.shap_values(X_sample_proc)

global_importance = (
    pd.DataFrame(np.abs(shap_values_sample), columns=feature_names_kor)
    .mean()
    .sort_values(ascending=False)
    .head(12)
    .reset_index()
)
global_importance.columns = ["변수", "평균 절대 기여도"]

fig_shap_global = px.bar(
    global_importance.sort_values("평균 절대 기여도"),
    x="평균 절대 기여도",
    y="변수",
    orientation="h",
    title="전체 환자 기준 SHAP 전역 중요도(평균 절대 기여도)",
)
fig_shap_global.update_layout(
    height=450,
    xaxis_title="평균 절대 기여도",
    yaxis_title="",
    margin=dict(l=20, r=20, t=60, b=20),
)
st.plotly_chart(fig_shap_global, use_container_width=True)


# -----------------------------
# (선택) 범주형 코드 확인
# -----------------------------
with st.expander("범주형 변수 코드 확인(데이터에 실제 존재하는 값)"):
    X0 = df.drop(columns=["target"])
    for c in art["cat_cols"]:
        st.write(f"- **{KOR_COL.get(c, c)} ({c})**: {sorted(pd.unique(X0[c]).tolist())}")


# -----------------------------
# 데이터 미리보기
# -----------------------------
with st.expander("데이터 미리보기(상위 20행)"):
    preview = df.copy()
    preview.columns = [KOR_COL.get(c, c) for c in preview.columns]
    st.dataframe(preview.head(20), use_container_width=True)
