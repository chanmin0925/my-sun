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


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜", "평균기온"])

    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# ==========================================
# 연도별 평균기온 계산
# ==========================================

yearly = (
    df.groupby("연도")
    .agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 2025년까지 사용
# 1년 관측일수가 300일 미만인 연도 제외
yearly = yearly[
    (yearly["연도"] <= 2025) &
    (yearly["관측일수"] >= 300)
].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)

# 회귀의 독립변수
# 1908년을 기준으로 몇 년이 지났는지
yearly["경과연수"] = yearly["연도"] - 1908


# ==========================================
# 전체 데이터 회귀
# ==========================================

st.subheader("1. 전체 데이터에 대한 선형회귀")

full_model = LinearRegression()

X_full = yearly[["경과연수"]]
y_full = yearly["평균기온"]

full_model.fit(X_full, y_full)

full_prediction = full_model.predict(X_full)

full_slope = full_model.coef_[0]
full_intercept = full_model.intercept_

full_mae = mean_absolute_error(y_full, full_prediction)
full_mse = mean_squared_error(y_full, full_prediction)
full_r2 = r2_score(y_full, full_prediction)

st.write(
    f"회귀식: "
    f"**평균기온 = {full_slope:.4f} × (연도 - 1908) + {full_intercept:.4f}**"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("기울기", f"{full_slope:.4f} °C/년")

with col2:
    st.metric("MAE", f"{full_mae:.3f} °C")

with col3:
    st.metric("MSE", f"{full_mse:.3f}")

with col4:
    st.metric("R²", f"{full_r2:.3f}")


# 전체 데이터 그래프
full_chart = yearly[["연도", "평균기온"]].copy()
full_chart = full_chart.set_index("연도")

st.line_chart(
    full_chart,
    y="평균기온",
    x_label="연도",
    y_label="평균기온 (°C)"
)

st.write(
    "이 그래프로 알 수 있는 것: "
    "전체 기간의 연평균 기온이 장기적으로 어떻게 변화해 왔는지 확인할 수 있습니다."
)


# ==========================================
# 학습 / 테스트 데이터
# ==========================================

st.subheader("2. 학습 데이터와 테스트 데이터")

st.write(
    "1956~2005년을 학습 데이터로 사용하는 최근 50년 모델과 "
    "1906~2005년을 학습 데이터로 사용하는 최근 100년 모델을 만들고, "
    "두 모델 모두 2006~2025년을 공통 테스트 데이터로 사용합니다."
)

train_50 = yearly[
    (yearly["연도"] >= 1956) &
    (yearly["연도"] <= 2005)
].copy()

train_100 = yearly[
    (yearly["연도"] >= 1906) &
    (yearly["연도"] <= 2005)
].copy()

test = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()


# ==========================================
# 최근 50년 모델
# ==========================================

model_50 = LinearRegression()

model_50.fit(
    train_50[["경과연수"]],
    train_50["평균기온"]
)

prediction_50 = model_50.predict(
    test[["경과연수"]]
)

slope_50 = model_50.coef_[0]
intercept_50 = model_50.intercept_

mae_50 = mean_absolute_error(
    test["평균기온"],
    prediction_50
)

mse_50 = mean_squared_error(
    test["평균기온"],
    prediction_50
)

r2_50 = r2_score(
    test["평균기온"],
    prediction_50
)


# ==========================================
# 최근 100년 모델
# ==========================================

model_100 = LinearRegression()

model_100.fit(
    train_100[["경과연수"]],
    train_100["평균기온"]
)

prediction_100 = model_100.predict(
    test[["경과연수"]]
)

slope_100 = model_100.coef_[0]
intercept_100 = model_100.intercept_

mae_100 = mean_absolute_error(
    test["평균기온"],
    prediction_100
)

mse_100 = mean_squared_error(
    test["평균기온"],
    prediction_100
)

r2_100 = r2_score(
    test["평균기온"],
    prediction_100
)


# ==========================================
# 모델 비교
# ==========================================

st.subheader("3. 최근 50년 vs 최근 100년 모델 비교")

comparison = pd.DataFrame({
    "모델": [
        "최근 50년 학습",
        "최근 100년 학습"
    ],
    "학습 기간": [
        "1956~2005",
        "1906~2005"
    ],
    "테스트 기간": [
        "2006~2025",
        "2006~2025"
    ],
    "기울기 (°C/년)": [
        slope_50,
        slope_100
    ],
    "MAE (°C)": [
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
        "MAE (°C)": "{:.3f}",
        "MSE": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 회귀선 비교
# ==========================================

st.subheader("4. 두 회귀선 비교")

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
    "학습에 사용한 기간에 따라 기온 상승 추세를 나타내는 회귀선의 기울기가 "
    "어떻게 달라지는지 확인할 수 있습니다."
)


# ==========================================
# 테스트 기간 실제값 + 예측값
# ==========================================

st.subheader("5. 2006~2025년 실제 기온과 예측값 비교")

test_chart = test[
    [
        "연도",
        "평균기온"
    ]
].copy()

test_chart["최근 50년 학습 예측"] = prediction_50
test_chart["최근 100년 학습 예측"] = prediction_100

test_chart = test_chart.set_index("연도")

st.line_chart(
    test_chart,
    x_label="연도",
    y_label="평균기온 (°C)"
)

st.write(
    "이 그래프로 알 수 있는 것: "
    "과거 데이터로 학습한 두 모델이 공통 테스트 기간인 2006~2025년의 "
    "실제 기온을 얼마나 잘 따라가는지 확인할 수 있습니다."
)


# ==========================================
# 테스트 데이터 상세 결과
# ==========================================

st.subheader("6. 테스트 데이터 예측 결과")

result = test[
    [
        "연도",
        "평균기온"
    ]
].copy()

result["50년 학습 예측"] = prediction_50
result["100년 학습 예측"] = prediction_100

result["50년 오차"] = (
    result["평균기온"] -
    result["50년 학습 예측"]
)

result["100년 오차"] = (
    result["평균기온"] -
    result["100년 학습 예측"]
)

st.dataframe(
    result.style.format({
        "평균기온": "{:.2f}",
        "50년 학습 예측": "{:.2f}",
        "100년 학습 예측": "{:.2f}",
        "50년 오차": "{:.2f}",
        "100년 오차": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 결과 해석
# ==========================================

st.subheader("7. 결과 해석")

st.write(
    f"**최근 50년 학습 모델의 기울기:** "
    f"{slope_50:.4f} °C/년"
)

st.write(
    f"**최근 100년 학습 모델의 기울기:** "
    f"{slope_100:.4f} °C/년"
)

slope_difference = slope_50 - slope_100

st.write(
    f"**두 모델의 기울기 차이:** "
    f"{slope_difference:.4f} °C/년"
)


if mae_50 < mae_100:
    mae_result = "최근 50년 모델의 MAE가 더 작아 평균적인 예측 오차가 더 작습니다."
elif mae_100 < mae_50:
    mae_result = "최근 100년 모델의 MAE가 더 작아 평균적인 예측 오차가 더 작습니다."
else:
    mae_result = "두 모델의 MAE가 같습니다."

if mse_50 < mse_100:
    mse_result = "최근 50년 모델의 MSE가 더 작아 큰 오차까지 고려했을 때 더 좋습니다."
elif mse_100 < mse_50:
    mse_result = "최근 100년 모델의 MSE가 더 작아 큰 오차까지 고려했을 때 더 좋습니다."
else:
    mse_result = "두 모델의 MSE가 같습니다."

if r2_50 > r2_100:
    r2_result = "최근 50년 모델의 R²가 더 높아 테스트 데이터의 변동을 더 잘 설명합니다."
elif r2_100 > r2_50:
    r2_result = "최근 100년 모델의 R²가 더 높아 테스트 데이터의 변동을 더 잘 설명합니다."
else:
    r2_result = "두 모델의 R²가 같습니다."

st.write(mae_result)
st.write(mse_result)
st.write(r2_result)


# ==========================================
# 미래 연도 예측
# ==========================================

st.subheader("8. 연도별 기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=int(yearly["연도"].min()),
    max_value=2050,
    value=2026,
    step=1
)

selected_x = pd.DataFrame({
    "경과연수": [selected_year - 1908]
})

future_50 = model_50.predict(selected_x)[0]
future_100 = model_100.predict(selected_x)[0]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "최근 50년 학습 모델",
        f"{future_50:.2f} °C"
    )

with col2:
    st.metric(
        "최근 100년 학습 모델",
        f"{future_100:.2f} °C"
    )

st.caption(
    "미래 예측값은 과거의 선형 추세를 미래로 단순 연장한 값입니다. "
    "실제 미래 기온을 보장하는 예측은 아닙니다."
)
