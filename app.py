import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# Title of the dashboard
st.title('Sample Streamlit Dashboard')

# Sidebar for user input
st.sidebar.header('User Input Parameters')
n = st.sidebar.slider('Number of data points', 10, 100, 50)
option = st.sidebar.selectbox('Select a chart type', ('Line Chart', 'Bar Chart'))

# Generate random data
data = pd.DataFrame({
    'x': np.arange(n),
    'y': np.random.randn(n).cumsum()
})

# Display the data
st.write('### Generated Data', data)

# Plot the data
st.write('### Chart')
if option == 'Line Chart':
    st.line_chart(data.set_index('x'))
elif option == 'Bar Chart':
    fig, ax = plt.subplots()
    ax.bar(data['x'], data['y'])
    st.pyplot(fig)

# Adding a map
st.write('### Map')
map_data = pd.DataFrame(
    np.random.randn(100, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])
st.map(map_data)

st.title('테스트중입니다.........')

# API 호출 함수
def fetch_data_from_api(url, api_key):
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # HTTP 에러가 발생하면 예외를 일으킴
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API 호출에 실패했습니다: {e}")
        return None

# Streamlit 앱 구성
st.title("API 데이터 가져오기")

api_url = st.text_input("API URL 입력", "https://api.example.com/data")

# API 키를 secrets에서 불러오기
api_key = st.secrets["api"]["key"]

if st.button("데이터 가져오기"):
    data = fetch_data_from_api(api_url, api_key)
    if data:
        # 데이터를 데이터프레임으로 변환
        df = pd.DataFrame(data)
        st.write(df)

gu_list = ["강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구", "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"]
url = "http://openAPI.seoul.go.kr:8088/4a496d74426a736d34327a6b6a4c55/json/CleanupBussinessInfo/1/25/금천구"
r = requests.get(url)
r.status_code
data = r.json()
cleanup = data['CleanupBussinessInfo']['row']
df = pd.DataFrame(cleanup)
cleanup_df = df[['CGG_NM', 'PRGRS_SEQ', 'STTS', 'USG_AREA']]
cleanup_df
