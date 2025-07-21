# app.py

import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load('salary_xgb_pipeline.pkl')

st.title('Employee Salary Prediction')

st.write('Fill all details:')

# === Required fields ===

age = st.number_input('Age', min_value=17, max_value=90, value=30)

workclass = st.selectbox('Workclass', [
    'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
    'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
])

fnlwgt = st.number_input('fnlwgt', min_value=10000, max_value=1000000, value=200000)

education = st.selectbox('Education', [
    'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
    'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th',
    'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th',
    'Preschool'
])

education_num = st.number_input('Education-num', min_value=1, max_value=16, value=10)

marital_status = st.selectbox('Marital Status', [
    'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
    'Widowed', 'Married-spouse-absent'
])

occupation = st.selectbox('Occupation', [
    'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
    'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
    'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
    'Transport-moving', 'Priv-house-serv', 'Protective-serv',
    'Armed-Forces'
])

relationship = st.selectbox('Relationship', [
    'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'
])

race = st.selectbox('Race', [
    'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'
])

sex = st.selectbox('Sex', ['Male', 'Female'])

capital_gain = st.number_input('Capital Gain', min_value=0, value=0)
capital_loss = st.number_input('Capital Loss', min_value=0, value=0)
hours_per_week = st.number_input('Hours per week', min_value=1, max_value=99, value=40)

native_country = st.selectbox('Native Country', [
    'United-States', 'Mexico', 'Philippines', 'Germany',
    'Canada', 'India', 'England', 'China', 'Cuba', 'Jamaica',
    'Italy', 'Puerto-Rico', 'South', 'Honduras', 'Other'
])

# === Predict ===

if st.button('Predict'):
    input_df = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [fnlwgt],
        'education': [education],
        'educational-num': [education_num],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'gender': [sex],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })

    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success('Predicted: Income >50K')
    else:
        st.warning('Predicted: Income <=50K')
