import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="기온 예측기", layout="wide")

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


# -----------------------------
# 데이터 불러오기
# -----------------------------
df = load_data()

# 2025년까지, 연간 관측일수가 300일 이상인 연도만 사용
yearly = (
    df.groupby("연도")
    .agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

yearly = yearly[
    (yearly["연도"] <= 2025) &
    (yearly["관측일수"] >= 300)
].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)

# 회귀의 독립변수: 1908년 이후 경과 연수
yearly["경과연수"] = yearly["연도"] - 1908

# -----------------------------
# 함수
# -----------------------------
def make_model(train_df):
    X = train_df[["경과연수"]]
    y = train_df["평균기온"]

    model = LinearRegression()
    model.fit(X, y)

    return model


def evaluate_model(model, test_df):
    X_test = test_df[["경과연수"]]
    y_test = test_df["평균기온"]

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    return mae, mse, r2, pred


# -----------------------------
# 전체 데이터 회귀
# -----------------------------
st.subheader("1. 전체 데이터로 만든 선형회귀")

full_data = yearly.copy()

full_model = make_model(full_data)

full_pred = full_model.predict(full_data[["경과연수"]])

full_mae = mean_absolute_error(full_data["평균기온"], full_pred)
full_mse = mean_squared_error(full_data["평균기온"], full_pred)
full_r2 = r2_score(full_data["평균기온"], full_pred)

full_slope = full_model.coef_[0]
full_intercept = full_model.intercept_

st.write(
    f"회귀식: **평균기온 = {full_slope:.4f} × (연도 - 1908) + {full_intercept:.4f}**"
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


fig_full = go.Figure()

fig_full.add_trace(
    go.Scatter(
        x=full_data["연도"],
        y=full_data["평균기온"],
        mode="markers",
        name="연평균 기온",
        hovertemplate="연도: %{x}<br>평균기온: %{y:.2f}°C<extra></extra>"
    )
)

line_x = full_data["연도"]
line_y = full_model.predict(full_data[["경과연수"]])

fig_full.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="전체 데이터 회귀선",
        hovertemplate="연도: %{x}<br>예측기온: %{y:.2f}°C<extra></extra>"
    )
)

fig_full.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified"
)

st.plotly_chart(fig_full, use_container_width=True)

st.write(
    "이 그래프로 알 수 있는 것: 전체 기간의 연평균 기온이 장기적으로 어느 방향으로 변화해 왔는지 확인할 수 있습니다."
)


# -----------------------------
# 학습 / 테스트 데이터 분리
# -----------------------------
st.subheader("2. 학습 데이터와 테스트 데이터 분리")

st.write(
    "1956~2005년 또는 1906~2005년의 과거 데이터를 학습하고, "
    "두 경우 모두 2006~2025년을 공통 테스트 데이터로 사용합니다."
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


# -----------------------------
# 50년 모델
# -----------------------------
model_50 = make_model(train_50)

mae_50, mse_50, r2_50, pred_50 = evaluate_model(model_50, test)

slope_50 = model_50.coef_[0]
intercept_50 = model_50.intercept_


# -----------------------------
# 100년 모델
# -----------------------------
model_100 = make_model(train_100)

mae_100, mse_100, r2_100, pred_100 = evaluate_model(model_100, test)

slope_100 = model_100.coef_[0]
intercept_100 = model_100.intercept_


# -----------------------------
# 성능 비교
# -----------------------------
st.subheader("3. 최근 50년 학습 vs 최근 100년 학습")

comparison = pd.DataFrame({
    "모델": [
        "최근 50년 학습 (1956~2005)",
        "최근 100년 학습 (1906~2005)"
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


# -----------------------------
# 회귀선 비교 그래프
# -----------------------------
st.subheader("4. 두 학습 기간의 회귀선 비교")

fig_compare = go.Figure()

fig_compare.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(size=6),
        hovertemplate="연도: %{x}<br>평균기온: %{y:.2f}°C<extra></extra>"
    )
)

# 50년 회귀선
line_years_50 = pd.DataFrame({
    "연도": list(range(1956, 2026))
})
line_years_50["경과연수"] = line_years_50["연도"] - 1908
line_years_50["예측기온"] = model_50.predict(
    line_years_50[["경과연수"]]
)

fig_compare.add_trace(
    go.Scatter(
        x=line_years_50["연도"],
        y=line_years_50["예측기온"],
        mode="lines",
        name="최근 50년 학습 회귀선",
        hovertemplate="연도: %{x}<br>예측기온: %{y:.2f}°C<extra></extra>"
    )
)

# 100년 회귀선
line_years_100 = pd.DataFrame({
    "연도": list(range(1906, 2026))
})
line_years_100["경과연수"] = line_years_100["연도"] - 1908
line_years_100["예측기온"] = model_100.predict(
    line_years_100[["경과연수"]]
)

fig_compare.add_trace(
    go.Scatter(
        x=line_years_100["연도"],
        y=line_years_100["예측기온"],
        mode="lines",
        name="최근 100년 학습 회귀선",
        hovertemplate="연도: %{x}<br>예측기온: %{y:.2f}°C<extra></extra>"
    )
)

# 테스트 기간 표시
fig_compare.add_vrect(
    x0=2006,
    x1=2025,
    fillcolor="gray",
    opacity=0.12,
    line_width=0,
    annotation_text="공통 테스트 기간",
    annotation_position="top left"
)

fig_compare.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified"
)

st.plotly_chart(fig_compare, use_container_width=True)

st.write(
    "이 그래프로 알 수 있는 것: 같은 2006~2025년을 예측하더라도 학습에 사용한 과거 기간에 따라 회귀선의 기울기와 예측값이 달라지는 것을 볼 수 있습니다."
)


# -----------------------------
# 테스트 데이터 예측 비교
# -----------------------------
st.subheader("5. 공통 테스트 데이터(2006~2025) 예측 비교")

test_result = test[["연도", "평균기온"]].copy()

test_result["50년 학습 예측"] = pred_50
test_result["100년 학습 예측"] = pred_100

test_result["50년 오차"] = (
    test_result["평균기온"] - test_result["50년 학습 예측"]
)

test_result["100년 오차"] = (
    test_result["평균기온"] - test_result["100년 학습 예측"]
)

st.dataframe(
    test_result.style.format({
        "평균기온": "{:.2f}",
        "50년 학습 예측": "{:.2f}",
        "100년 학습 예측": "{:.2f}",
        "50년 오차": "{:.2f}",
        "100년 오차": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# -----------------------------
# 예측값 그래프
# -----------------------------
fig_test = go.Figure()

fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="lines+markers",
        name="실제 평균기온",
        hovertemplate="연도: %{x}<br>실제: %{y:.2f}°C<extra></extra>"
    )
)

fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines+markers",
        name="50년 학습 예측",
        hovertemplate="연도: %{x}<br>50년 예측: %{y:.2f}°C<extra></extra>"
    )
)

fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_100,
        mode="lines+markers",
        name="100년 학습 예측",
        hovertemplate="연도: %{x}<br>100년 예측: %{y:.2f}°C<extra></extra>"
    )
)

fig_test.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified"
)

st.plotly_chart(fig_test, use_container_width=True)

st.write(
    "이 그래프로 알 수 있는 것: 과거 데이터로 학습한 두 회귀모델이 실제 2006~2025년의 기온 변화를 얼마나 잘 따라가는지 비교할 수 있습니다."
)


# -----------------------------
# 결과 해석
# -----------------------------
st.subheader("6. 결과 비교")

slope_diff = slope_50 - slope_100

if slope_diff > 0:
    slope_message = (
        f"최근 50년 모델의 기울기({slope_50:.4f} °C/년)가 "
        f"최근 100년 모델({slope_100:.4f} °C/년)보다 큽니다. "
        "이는 비교적 최근의 기온 상승 추세가 더 가파르게 나타난다는 뜻입니다."
    )
elif slope_diff < 0:
    slope_message = (
        f"최근 100년 모델의 기울기({slope_100:.4f} °C/년)가 "
        f"최근 50년 모델({slope_50:.4f} °C/년)보다 큽니다. "
        "즉, 장기간의 자료를 사용했을 때 상승 추세가 더 가파르게 나타납니다."
    )
else:
    slope_message = "두 모델의 기울기가 거의 같습니다."

st.write(slope_message)

# 어떤 모델이 더 좋은지 판단
# MAE와 MSE는 작을수록 좋고, R²는 클수록 좋음
score_50 = 0
score_100 = 0

if mae_50 < mae_100:
    score_50 += 1
elif mae_100 < mae_50:
    score_100 += 1

if mse_50 < mse_100:
    score_50 += 1
elif mse_100 < mse_50:
    score_100 += 1

if r2_50 > r2_100:
    score_50 += 1
elif r2_100 > r2_50:
    score_100 += 1

if score_50 > score_100:
    better_model = "최근 50년 학습 모델"
elif score_100 > score_50:
    better_model = "최근 100년 학습 모델"
else:
    better_model = "두 모델이 비슷함"

st.info(
    f"공통 테스트 기간(2006~2025)의 예측 성능을 MAE, MSE, R²로 비교하면 "
    f"**{better_model}**이 더 좋은 성능을 보입니다."
)

st.write(
    "MAE는 실제 기온과 예측 기온의 절대적인 차이의 평균으로 작을수록 좋고, "
    "MSE는 오차를 제곱하여 평균한 값으로 역시 작을수록 좋습니다. "
    "R²는 모델이 실제 기온의 변동을 얼마나 설명하는지를 나타내며 일반적으로 클수록 좋습니다. "
    "R²는 테스트 데이터에서 0보다 작아질 수도 있습니다."
)

# -----------------------------
# 예측 슬라이더
# -----------------------------
st.subheader("7. 연도별 기온 예측")

min_year = int(yearly["연도"].min())
max_year = 2050

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=min_year,
    max_value=max_year,
    value=2026,
    step=1
)

selected_x = [[selected_year - 1908]]

prediction_50 = model_50.predict(selected_x)[0]
prediction_100 = model_100.predict(selected_x)[0]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "최근 50년 학습 모델",
        f"{prediction_50:.2f} °C"
    )

with col2:
    st.metric(
        "최근 100년 학습 모델",
        f"{prediction_100:.2f} °C"
    )

st.caption(
    "위 예측값은 선형회귀선을 미래 연도까지 단순히 연장한 값이며, "
    "실제 미래 기온을 보장하는 값은 아닙니다."
)
