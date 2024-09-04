import streamlit as st
import numpy as np


st.title("투수 및 타자 연봉 예측 모델🤖")


with st.chat_message("assistant"):
    
    st.write("""
이 웹은 메이저리그 투수와 타자의 다양한 경기 성적 데이터를 기반으로 연봉을 예측하는 모델입니다.
데이터를 탐색하고, 모델을 학습시키고, 새 데이터를 통해 예측해볼 수 있습니다.
""")


import matplotlib.pyplot as plt
from scipy.stats import norm

# 데이터 생성 (정규분포)
mu, sigma = 0, 1  # 평균과 표준편차
data = np.random.normal(mu, sigma, 1000)

# 1분위에 해당하는 값 계산 (하위 25%)
quantile_25 = np.percentile(data, 25)

# 정규분포 커브 그리기
x = np.linspace(min(data), max(data), 1000)
y = norm.pdf(x, mu, sigma)

# 그래프 그리기
plt.figure(figsize=(10, 6))

# 전체 정규분포 그래프
plt.plot(x, y, label='Normal Distribution')

# 1분위 구간 색칠
x_fill = np.linspace(min(data), quantile_25, 100)
y_fill = norm.pdf(x_fill, mu, sigma)
plt.fill_between(x_fill, y_fill, color='skyblue', alpha=0.6, label='1st Quartile (0-25%)')

# 1분위 구간을 강조하는 수직선
plt.axvline(quantile_25, color='red', linestyle='--', label='25th Percentile')

# 레이블 및 범례 추가
plt.title('Normal Distribution with 1st Quartile Shaded')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# 그래프 출력
plt.show()
