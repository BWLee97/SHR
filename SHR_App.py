import numpy as np
import pandas as pd
import streamlit as st
import requests
import os
import pickle

st.set_page_config(layout="wide")
st.title('28天全因死亡率预测')

with st.spinner("Loading..."):
    file_path = "model.pkl"
    if os.path.exists(file_path):
        st.success("Model downloaded successfully.", icon="✅")
    else:
        nomogram_url = "https://gitee.com/LeeBoWen/SHR/raw/master/model.pkl"
        response = requests.get(nomogram_url)
        with open('model.pkl', 'wb') as file:
            file.write(response.content)
        
        st.success("Model downloaded successfully.", icon="✅")

if os.path.exists('model.pkl'):
    input_data = {'Charlson comorbidity index': 0.0, 'Age': 50.0,
                  'PT': 12.0, 'INR': 1.0, 'Severe liver disease': 0,
                  'Temperature': 36.5, 'Na+': 140.0, 'SAPS II': 30.0,
                  'RBC': 4.5, 'CV': 0, 'BUN': 5.0, 'MBP': 80.0,
                  'Aniongap': 10.0, 'DBP': 70.0, 'HCO3-': 24.0,
                  'Glucose': 5.5, 'SBP': 120.0, 'Cl-': 100.0,
                  'Hematocrit': 40.0, 'Lactate': 1.0}
    # 准备提交表单
    with st.form('my_form'):
        st.subheader('Please enter the following clinical indicators for prediction.')
        col_1, col_2, col_3, col_4 = st.columns(4)
        with col_1:
            input_data['Charlson comorbidity index'] = st.number_input('Charlson comorbidity index', 0.0, 13.0, step=0.1)
            input_data['Age'] = st.number_input('Age', 19.0, 99.0, step=0.1)
            input_data['PT'] = st.number_input('PT', 8.0, 145.0, step=0.1)
            input_data['INR'] = st.number_input('INR', 0.7, 13.0, step=0.1)
            SLD = st.selectbox('Severe liver disease', options=['No', 'Yes'])
            input_data['Severe liver disease'] = 1 if SLD == 'Yes' else 0
        with col_2:
            input_data['Temperature'] = st.number_input('Temperature', 33.0, 40.0, step=0.1)
            input_data['Na+'] = st.number_input('Na+', 107.0, 159.0, step=0.1)
            input_data['SAPS II'] = st.number_input('SAPS II', 11.0, 152.0, step=0.1)
            input_data['RBC'] = st.number_input('RBC', 1.3, 6.1, step=0.1)
            input_data['CV'] = st.number_input('CV', 1.2, 144.0, step=0.1)
        with col_3:
            input_data['BUN'] = st.number_input('BUN', 2.0, 145.0, step=0.1)
            input_data['MBP'] = st.number_input('MBP', 50.0, 130.0, step=0.1)
            input_data['Aniongap'] = st.number_input('Aniongap', 5.5, 32.0, step=0.1)
            input_data['DBP'] = st.number_input('DBP', 55.0, 106.0, step=0.1)
            input_data['HCO3-'] = st.number_input('HCO3-', 19.0, 42.0, step=0.1)
        with col_4:
            input_data['Glucose'] = st.number_input('Glucose', 52.5, 419.5, step=0.1)
            input_data['SBP'] = st.number_input('SBP', 72.0, 179.0, step=0.1)
            input_data['Cl-'] = st.number_input('Cl-', 75.0, 132.0, step=0.1)
            input_data['Hematocrit'] = st.number_input('Hematocrit', 13.0, 55.0, step=0.1)
            input_data['Lactate'] = st.number_input('Lactate', 0.0, 12.5, step=0.1)

        submitted = st.form_submit_button('Predict')
else:
    st.error("Model download failed, please email the author.", icon="🚨")

with st.container(border=True):
    st.subheader('Prediction Result')
    if submitted:

        def norm(x, xmin, xmax):
            x = (x - xmin)/(xmax-xmin)
            return x

        input_data['Charlson comorbidity index'] = norm(input_data['Charlson comorbidity index'], 0, 13)
        input_data['Age'] = norm(input_data['Age'], 18.10, 98.65)
        input_data['PT'] = norm(input_data['PT'], 8.7, 144.63)
        input_data['INR'] = norm(input_data['INR'], 0.7, 12.63)
        input_data['Temperature'] = norm(input_data['Temperature'], 33.19, 39.67)
        input_data['Na+'] = norm(input_data['Na+'], 107.33, 158.75)
        input_data['SAPS II'] = norm(input_data['SAPS II'], 11, 152)
        input_data['RBC'] = norm(input_data['RBC'], 1.33, 6.12)
        input_data['CV'] = norm(input_data['CV'], 1.25, 143.73)
        input_data['BUN'] = norm(input_data['BUN'], 2, 145.33)
        input_data['MBP'] = norm(input_data['MBP'], 50.69, 130.19)
        input_data['Aniongap'] = norm(input_data['Aniongap'], 5.5, 32.33)
        input_data['DBP'] = norm(input_data['DBP'], 55.59, 105.83)
        input_data['HCO3-'] = norm(input_data['HCO3-'], 9, 41.33)
        input_data['Glucose'] = norm(input_data['Glucose'], 52.5, 419.5)
        input_data['SBP'] = norm(input_data['SBP'], 72.64, 178.93)
        input_data['Cl-'] = norm(input_data['Cl-'], 75.17, 132)
        input_data['Hematocrit'] = norm(input_data['Hematocrit'], 13, 54.6)
        input_data['Lactate'] = norm(input_data['Lactate'], -0.1, 12.46)
        input_data = pd.DataFrame([input_data])

        model = pickle.load(open(file_path, 'rb'))
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1]
        probability = f"{probability[0] * 100:.2f} %"

        if prediction[0] == 0:
            st.info(f'**This patient has a low death risk with probability of {probability}.**', icon="ℹ️")
        else:
            st.error(f'**This patient is at high death risk with probability of {probability}!**', icon="🚨")
    else:
        st.warning('Please enter the relevant information and then press **Predict**.', icon="⚠️")

