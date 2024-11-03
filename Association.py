import pandas as pd
import streamlit as st

data = pd.read_csv('https://raw.githubusercontent.com/kmrhrsid/jie43203/refs/heads/main/Crop_recommendation.csv')

st.write(data)
