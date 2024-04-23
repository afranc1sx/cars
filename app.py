import streamlit as st
from joblib import load
import numpy as np
import os

#load trained model
def load_model():
    return load('C:/Users/a6shl/OneDrive/Documents/GitHub/carzz/knn_model.joblib')

# title 
st.title('Connected Vehicle Cyberattack Simulation')

id_input = st.number_input('Enter ID', value=0)
dlc_input = st.number_input('Enter DLC', value=0)
data0_input = st.number_input('Enter Data0', value=0)
data1_input = st.number_input('Enter Data1', value=0)
data2_input = st.number_input('Enter Data2', value=0)
data3_input = st.number_input('Enter Data3', value=0)
data4_input = st.number_input('Enter Data4', value=0)
data5_input = st.number_input('Enter Data5', value=0)
data6_input = st.number_input('Enter Data6', value=0)
data7_input = st.number_input('Enter Data7', value=0)

#create array to store the data
input_data = np.array([id_input, dlc_input, data0_input, data1_input, 
                       data2_input, data3_input, data4_input, data5_input,
                       data6_input, data7_input]).reshape(1, -1)

prediction = load_model.predict(input_data)

