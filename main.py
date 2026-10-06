import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="기온 예측기",
    layout="wide"
)

st.title("기온 예측기")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"


# =========================================================
# 데이터 불러오기
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜", "평균기온"])

    df["연도"] = df["날짜"].dt.year

    # 연도별 평균기온과 관측일수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 2025년까지 사용
    # 관측일수가 300일 미만인 연도는 제외
    yearly = yearly[
        (yearly["연도"] <= 2025) &
        (yearly["관측일수"] >= 300)
    ].copy()

    yearly = yearly.sort_values("연도").reset_index(drop=True)

    # 회귀에 사용할 독립변수
    # 1908년부터 몇 년이 지났는지
    yearly["경과연수"] = yearly["연도"] - 1908

    return yearly


yearly = load_data()


# =========================================================
# 1. 전체 데이터에 대한 선형회귀
# =========================================================

st.header("1. 전체 데이터에 대한 선형회귀")

st.write(
    "사용 가능한 전체 연평균 기온 데이터를 하나의 선형회귀 모델로 학습합니다."
)

X_all = yearly[["경과연수"]]
y_all = yearly["평균기온"]

model_all = LinearRegression()
model_all.fit(X_all, y_all)

pred_all = model_all.predict(X_all)

slope_all = model_all.coef_[0]
intercept_all = model_all.intercept_

mae_all = mean_absolute_error(y_all, pred_all)
mse_all = mean_squared_error(y_all, pred_all)
r2_all = r2_score(y_all, pred_all)

st.write(
    f"회귀식: "
    f"**평균기온 = {slope_all:.4f} × (연도 - 1908) + {intercept_all:.4f}**"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("기울기", f"{slope_all:.4f} °C/년")

with col2:
    st.metric("MAE", f"{mae_all:.3f} °C")

with col3:
    st.metric("MSE", f"{mse_all:.3f}")

with col4:
    st.metric("R²", f"{r2_all:.3f}")


# 전체 데이터 실제값
all_chart = yearly[
    ["연도", "평균기온"]
].copy()

all_chart = all_chart.set_index("연도")

st.line_chart(
    all_chart,
    y="평균기온",
    x_label="연도",
    y_label="평균기온 (°C)"
)

st.write(
    "이 그래프로 알 수 있는 것: "
    "전체 기간의 연평균 기온이 장기적으로 증가하는지 감소하는지 확인할 수 있습니다."
)


# =========================================================
# 2. 학습 데이터와 테스트 데이터 분리
# =========================================================

st.header("2. 학습 데이터와 테스트 데이터")

# 최근 50년 학습 데이터
train_50 = yearly[
    (yearly["연도"] >= 1956) &
    (yearly["연도"] <= 2005)
].copy()

# 최근 100년 학습 데이터
train_100 = yearly[
    (yearly["연도"] >= 1906) &
    (yearly["연도"] <= 2005)
].copy()

# 두 모델이 공통으로 사용하는 테스트 데이터
test = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()

st.write(
    f"최근 50년 학습 데이터: **{train_50['연도'].min()}~{train_50['연도'].max()}년** "
    f"({len(train_50)}개 연도)"
)

st.write(
    f"최근 100년 학습 데이터: **{train_100['연도'].min()}~{train_100['연도'].max()}년** "
    f"({len(train_100)}개 연도)"
)

st.write(
    f"공통 테스트 데이터: **{test['연도'].min()}~{test['연도'].max()}년** "
    f"({len(test)}개 연도)"
)


# =========================================================
# 3. 최근 50년 학습 모델
# =========================================================

model_50 = LinearRegression()

model_50.fit(
    train_50[["경과연수"]],
    train_50["평균기온"]
)

pred_50 = model_50.predict(
    test[["경과연수"]]
)

slope_50 = model_50.coef_[0]
intercept_50 = model_50.intercept_

mae_50 = mean_absolute_error(
    test["평균기온"],
    pred_50
)

mse_50 = mean_squared_error(
    test["평균기온"],
    pred_50
)

r2_50 = r2_score(
    test["평균기온"],
    pred_50
)


# =========================================================
# 4. 최근 100년 학습 모델
# =========================================================

model_100 = LinearRegression()

model_100.fit(
    train_100[["경과연수"]],
    train_100["평균기온"]
)

pred_100 = model_100.predict(
    test[["경과연수"]]
)

slope_100 = model_100.coef_[0]
intercept_100 = model_100.intercept_

mae_100 = mean_absolute_error(
    test["평균기온"],
    pred_100
)

mse_100 = mean_squared_error(
    test["평균기온"],
    pred_100
)

r2_100 = r2_score(
    test["평균기온"],
    pred_100
)


# =========================================================
# 5. 50년 / 100년 모델 성능 비교
# =========================================================

st.header("3. 최근 50년 vs 최근 100년 학습 모델")

st.write(
    "두 모델 모두 **2006~2025년을 한 번도 학습하지 않은 공통 테스트 데이터**로 평가합니다."
)

comparison = pd.DataFrame({
    "모델": [
        "최근 50년 모델",
        "최근 100년 모델"
    ],
    "훈련 데이터": [
        "1956~2005",
        "1906~2005"
    ],
    "테스트 데이터": [
        "2006~2025",
        "2006~2025"
    ],
    "기울기 (°C/년)": [
        slope_50,
        slope_100
    ],
    "MAE": [
        mae_50,
        mae_100
    ],
    "MSE": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})

st.dataframe(
    comparison.style.format({
        "기울기 (°C/년)": "{:.4f}",
        "MAE": "{:.3f}",
        "MSE": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 6. 기울기 비교
# =========================================================

st.subheader("회귀선의 기울기 비교")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "최근 50년 기울기",
        f"{slope_50:.4f} °C/년"
    )

with col2:
    st.metric(
        "최근 100년 기울기",
        f"{slope_100:.4f} °C/년"
    )

slope_difference = slope_50 - slope_100

st.write(
    f"두 회귀선의 기울기 차이: **{slope_difference:.4f} °C/년**"
)

if slope_50 > slope_100:
    st.info(
        "최근 50년 데이터로 학습한 회귀선의 기울기가 더 큽니다. "
        "즉, 1956~2005년의 추세를 이용했을 때 장기적인 기온 상승 속도가 "
        "1906~2005년 전체를 이용했을 때보다 더 크게 나타납니다."
    )
elif slope_50 < slope_100:
    st.info(
        "최근 100년 데이터로 학습한 회귀선의 기울기가 더 큽니다. "
        "즉, 1906~2005년의 장기 추세가 최근 50년의 추세보다 더 큰 기울기를 보입니다."
    )
else:
    st.info(
        "두 학습 기간의 회귀선 기울기가 같습니다."
    )


# =========================================================
# 7. 두 회귀선 비교
# =========================================================

st.subheader("두 학습 기간의 회귀선")

line_years = pd.DataFrame({
    "연도": range(1906, 2026)
})

line_years["경과연수"] = line_years["연도"] - 1908

line_years["최근 50년 회귀선"] = model_50.predict(
    line_years[["경과연수"]]
)

line_years["최근 100년 회귀선"] = model_100.predict(
    line_years[["경과연수"]]
)

line_chart = line_years.set_index("연도")

st.line_chart(
    line_chart[
        [
            "최근 50년 회귀선",
            "최근 100년 회귀선"
        ]
    ],
    x_label="연도",
    y_label="예측 평균기온 (°C)"
)

st.write(
    "이 그래프로 알 수 있는 것: "
    "훈련 데이터의 기간에 따라 만들어지는 기온 상승 추세선의 차이를 확인할 수 있습니다."
)


# =========================================================
# 8. 공통 테스트 데이터에서 실제값과 예측값 비교
# =========================================================

st.header("4. 2006~2025년 테스트 데이터 예측")

test_result = test[
    ["연도", "평균기온"]
].copy()

test_result["최근 50년 예측"] = pred_50
test_result["최근 100년 예측"] = pred_100

test_chart = test_result.set_index("연도")

st.line_chart(
    test_chart[
        [
            "평균기온",
            "최근 50년 예측",
            "최근 100년 예측"
        ]
    ],
    x_label="연도",
    y_label="평균기온 (°C)"
)

st.write(
    "이 그래프로 알 수 있는 것: "
    "1956~2005년 또는 1906~2005년만 사용해 학습한 회귀선이 "
    "한 번도 보지 못한 2006~2025년의 실제 기온을 얼마나 잘 따라가는지 확인할 수 있습니다."
)


# =========================================================
# 9. MAE / MSE / R² 해석
# =========================================================

st.header("5. 테스트 성능 평가")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("MAE")
    st.write(
        "실제 기온과 예측 기온의 평균적인 절대 오차입니다. "
        "작을수록 예측이 정확합니다."
    )

with col2:
    st.subheader("MSE")
    st.write(
        "예측 오차를 제곱한 뒤 평균한 값입니다. "
        "큰 오차에 더 큰 벌점을 줍니다. 작을수록 좋습니다."
    )

with col3:
    st.subheader("R²")
    st.write(
        "모델이 테스트 데이터의 기온 변화를 얼마나 설명하는지를 나타냅니다. "
        "일반적으로 클수록 좋습니다."
    )


# MAE 비교
if mae_50 < mae_100:
    mae_winner = "최근 50년 모델"
elif mae_100 < mae_50:
    mae_winner = "최근 100년 모델"
else:
    mae_winner = "두 모델 동일"

# MSE 비교
if mse_50 < mse_100:
    mse_winner = "최근 50년 모델"
elif mse_100 < mse_50:
    mse_winner = "최근 100년 모델"
else:
    mse_winner = "두 모델 동일"

# R² 비교
if r2_50 > r2_100:
    r2_winner = "최근 50년 모델"
elif r2_100 > r2_50:
    r2_winner = "최근 100년 모델"
else:
    r2_winner = "두 모델 동일"


st.write(f"**MAE가 더 좋은 모델:** {mae_winner}")
st.write(f"**MSE가 더 좋은 모델:** {mse_winner}")
st.write(f"**R²가 더 좋은 모델:** {r2_winner}")


# =========================================================
# 10. 테스트 성능 한눈에 비교
# =========================================================

st.header("6. 예측 성능 한눈에 비교")

metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    st.subheader("최근 50년 학습")

    st.metric(
        "MAE",
        f"{mae_50:.3f} °C"
    )

    st.metric(
        "MSE",
        f"{mse_50:.3f}"
    )

    st.metric(
        "R²",
        f"{r2_50:.3f}"
    )

with metric_col2:
    st.subheader("최근 100년 학습")

    st.metric(
        "MAE",
        f"{mae_100:.3f} °C"
    )

    st.metric(
        "MSE",
        f"{mse_100:.3f}"
    )

    st.metric(
        "R²",
        f"{r2_100:.3f}"
    )


# =========================================================
# 11. 최종 요약
# =========================================================

st.header("7. 최종 비교")

st.write(
    f"""
**최근 50년 학습**
- 훈련: 1956~2005년
- 테스트: 2006~2025년
- 회귀선 기울기: {slope_50:.4f} °C/년
- MAE: {mae_50:.3f} °C
- MSE: {mse_50:.3f}
- R²: {r2_50:.3f}

**최근 100년 학습**
- 훈련: 1906~2005년
- 테스트: 2006~2025년
- 회귀선 기울기: {slope_100:.4f} °C/년
- MAE: {mae_100:.3f} °C
- MSE: {mse_100:.3f}
- R²: {r2_100:.3f}
"""
)

st.success(
    "이 분석에서는 테스트 데이터인 2006~2025년을 학습 과정에서 제외했기 때문에, "
    "두 회귀모델의 실제 예측 성능을 공정하게 비교할 수 있습니다."
)
