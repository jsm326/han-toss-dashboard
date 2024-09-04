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

# 데이터 예시
data = sns.load_dataset('diamonds')['price']

# 경험적 누적 분포 함수 (ECDF) 플롯 생성
fig, ax = plt.subplots()
sns.ecdfplot(data, ax=ax)

# 특정 값(상위 퍼센트)의 표시
selected_value = data.quantile(0.95)  # 상위 5%에 해당하는 값
ax.axvline(selected_value, color='red', linestyle='--', label='상위 5%')
ax.legend()

st.pyplot(fig)


