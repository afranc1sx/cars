import streamlit as st
from joblib import load
import numpy as np
import os



# Print the current working directory
print("Current working directory:", os.getcwd())

# Load the trained model
@st.cache
def load_model():
    return load('C:/Users/a6shl/OneDrive/Documents/GitHub/carzz/cars/knn_model.joblib')  # Specify the full path to the model file

# Load the model
model = load_model()

# Streamlit app title
st.title('Vehicle Cyberattack Detection')

# Input fields for user to enter features
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

# Prepare input features for prediction
input_data = np.array([id_input, dlc_input, data0_input, data1_input, 
                       data2_input, data3_input, data4_input, data5_input,
                       data6_input, data7_input]).reshape(1, -1)

# Make prediction
prediction = model.predict(input_data)

# Display prediction
if prediction == 0:
    st.write('No cyberattack detected.')
else:
    st.write('Cyberattack detected!')
