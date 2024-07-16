import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 데이터 로드
file_path = "./data/ecos.csv"

df = pd.read_csv(file_path)
df['DATE'] = pd.to_datetime(df['DATE'])
st.write(df)

# Streamlit 앱 제목
st.title('Ecos Economic Data Analysis')

# 데이터프레임 표시
st.write("## DataFrame Head")
st.write(df.head())

# 년도 설정
years = mdates.YearLocator()  # every year
years_fmt = mdates.DateFormatter('%Y')

# 그래프 그리기
fig, ax = plt.subplots(3, 2, figsize=(20, 15))
# 첫 번째 서브플롯
ax[0, 0].plot(df['DATE'], df['aptpriceindex'])
ax[0, 0].set_xlabel('Date')
ax[0, 0].set_ylabel('Price Index')
ax[0, 0].set_title('Price Index Over Time')
ax[0, 0].xaxis.set_major_locator(years)
ax[0, 0].xaxis.set_major_formatter(years_fmt)

ax[0, 1].plot(df['DATE'], df['cd'])
ax[0, 1].set_xlabel('Date')
ax[0, 1].set_ylabel('CD')
ax[0, 1].set_title('CD Over Time')

ax[1, 0].plot(df['DATE'], df['unemploymentrate'])
ax[1, 0].set_xlabel('Date')
ax[1, 0].set_ylabel('unemploymentrate')
ax[1, 0].set_title('unemploymentrate Over Time')

ax[1, 1].plot(df['DATE'], df['constructionpricerate'])
ax[1, 1].set_xlabel('Date')
ax[1, 1].set_ylabel('constructionpricerate')
ax[1, 1].set_title('constructionpricerate Over Time')

ax[2, 0].plot(df['DATE'], df['cpi'])
ax[2, 0].set_xlabel('Date')
ax[2, 0].set_ylabel('cpi')
ax[2, 0].set_title('cpi Over Time')

# 다섯 가지 지표를 한 그래프에 표시
ax[2, 1].plot(df['DATE'], df['aptpriceindex'], marker='o', label='aptpriceindex')
ax[2, 1].plot(df['DATE'], df['cd'], marker='x', label='cd')
ax[2, 1].plot(df['DATE'], df['unemploymentrate'], marker='s', label='unemploymentrate')
ax[2, 1].plot(df['DATE'], df['constructionpricerate'], marker='d', label='constructionpricerate')
ax[2, 1].plot(df['DATE'], df['cpi'], marker='^', label='cpi')
ax[2, 1].set_xlabel('Date')
ax[2, 1].set_ylabel('Value')
ax[2, 1].set_title('All Indicators Over Time')
ax[2, 1].set_yscale('log')
ax[2, 1].legend()

# 날짜 레이블이 잘 보이도록 회전
for ax_row in ax:
    for ax_item in ax_row:
        for label in ax_item.get_xticklabels():
            label.set_rotation(45)

# 레이아웃 조정
plt.tight_layout()

# Streamlit을 통해 그래프 표시
st.pyplot(fig)

# 회귀 분석
X = df[['cd', 'unemploymentrate', 'constructionpricerate', 'cpi']]
y = df['aptpriceindex']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# 회귀 분석 결과 표로 출력
st.write("## Regression Analysis Summary")
st.write(model.summary().tables[1].as_html(), unsafe_allow_html=True)

# 잔차 플롯
fig_resid = plt.figure(figsize=(10, 6))
sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, line_kws={'color': 'red'})
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
st.pyplot(fig_resid)

# 잔차의 정규성 검정 (Q-Q Plot)
fig_qq = plt.figure()
sm.qqplot(model.resid, line='s')
plt.title('Normal Q-Q')
st.pyplot(fig_qq)

# 예측 값 대 실제 값 플롯
fig_fitted = plt.figure(figsize=(10, 6))
plt.scatter(model.fittedvalues, y)
plt.plot(model.fittedvalues, model.fittedvalues, color='red')
plt.xlabel('Fitted values')
plt.ylabel('Actual values')
plt.title('Fitted vs Actual')
st.pyplot(fig_fitted)

# 잔차의 자기상관 (ACF 플롯)
fig_acf, ax_acf = plt.subplots(1, 2, figsize=(16, 6))
plot_acf(model.resid, lags=30, ax=ax_acf[0])
plot_pacf(model.resid, lags=30, ax=ax_acf[1])
ax_acf[0].set_title('ACF of Residuals')
ax_acf[1].set_title('PACF of Residuals')
st.pyplot(fig_acf)
