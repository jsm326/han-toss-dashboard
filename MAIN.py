import streamlit as st
import numpy as np


st.title("투수 및 타자 연봉 예측 모델🤖")


with st.chat_message("assistant"):
    
    st.write("""
이 웹은 메이저리그 투수와 타자의 다양한 경기 성적 데이터를 기반으로 연봉을 예측하는 모델입니다.
데이터를 탐색하고, 모델을 학습시키고, 새 데이터를 통해 예측해볼 수 있습니다.
""")


import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 생성
data = sns.load_dataset('diamonds')['price']

# 상위 25%에 해당하는 값을 계산
quantile_75 = data.quantile(0.75)

# ECDF 플롯 생성
fig, ax = plt.subplots()

# 전체 구간을 시각화
sns.ecdfplot(data, ax=ax, label='ECDF', color='blue')

# 상위 25% 구간에 해당하는 데이터 필터링
high_data = data[data >= quantile_75]

# 상위 25% 구간을 따로 시각화 (다른 색상)
sns.ecdfplot(high_data, ax=ax, color='red')

# 상위 25% 라인 표시
ax.axvline(quantile_75, color='green', linestyle='--', label='상위 25% 시작점')

# 범례 추가
ax.legend()

# 제목 및 축 레이블 설정
ax.set_title('ECDF with Top 25% Highlighted')
ax.set_xlabel('Price')
ax.set_ylabel('Cumulative Probability')

# Streamlit에서 그래프 출력
st.pyplot(fig)


