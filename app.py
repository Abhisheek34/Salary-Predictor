
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

# Load saved LabelEncoders
workclass_encoder = joblib.load("workclass_encoder.pkl")
marital_status_encoder = joblib.load("marital-status_encoder.pkl")
occupation_encoder = joblib.load("occupation_encoder.pkl")
relationship_encoder = joblib.load("relationship_encoder.pkl")
race_encoder = joblib.load("race_encoder.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
native_country_encoder = joblib.load("native-country_encoder.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")

# Numeric Inputs
age = st.sidebar.slider("Age", 18, 90, 30)
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=1000, max_value=1000000, value=100000)
edu_num = st.sidebar.slider("Educational Number", 1, 16, 10)
capital_gain = st.sidebar.number_input("Capital Gain", value=0)
capital_loss = st.sidebar.number_input("Capital Loss", value=0)
hours_per_week = st.sidebar.slider("Hours Per Week", 1, 100, 40)

# Categorical Inputs with encoded classes
workclass = st.sidebar.selectbox("Workclass", workclass_encoder.classes_)
marital_status = st.sidebar.selectbox("Marital Status", marital_status_encoder.classes_)
occupation = st.sidebar.selectbox("Occupation", occupation_encoder.classes_)
relationship = st.sidebar.selectbox("Relationship", relationship_encoder.classes_)
race = st.sidebar.selectbox("Race", race_encoder.classes_)
gender = st.sidebar.selectbox("Gender", gender_encoder.classes_)
native_country = st.sidebar.selectbox("Native Country", native_country_encoder.classes_)

# Apply label encoding
input_data = {
    'age': age,
    'workclass': workclass_encoder.transform([workclass])[0],
    'fnlwgt': fnlwgt,
    'educational-num': edu_num,
    'marital-status': marital_status_encoder.transform([marital_status])[0],
    'occupation': occupation_encoder.transform([occupation])[0],
    'relationship': relationship_encoder.transform([relationship])[0],
    'race': race_encoder.transform([race])[0],
    'gender': gender_encoder.transform([gender])[0],
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country_encoder.transform([native_country])[0]
}

input_df = pd.DataFrame([input_data])

st.write("### 🔎 Input Data Preview")
st.write(input_df)

# Predict on user input
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    label_map = {0: "≤50K", 1: ">50K"}
    st.success(f"✅ Prediction: {prediction[0]}")

'''
# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    
    # Assuming prediction is like ['<=50K']
    if prediction[0] == "<=50K":
        st.success("✅ Prediction: Salary is ≤ 50K")
    else:
        st.success("✅ Prediction: Salary is > 50K")

'''

#if prediction[0] == "<=50K":
#   st.success("✅ Prediction: Salary is ≤ 50K")
#else:
#   st.success("✅ Prediction: Salary is > 50K")


# Batch prediction section
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    # Apply encoders
    batch_data['workclass'] = workclass_encoder.transform(batch_data['workclass'])
    batch_data['marital-status'] = marital_status_encoder.transform(batch_data['marital-status'])
    batch_data['occupation'] = occupation_encoder.transform(batch_data['occupation'])
    batch_data['relationship'] = relationship_encoder.transform(batch_data['relationship'])
    batch_data['race'] = race_encoder.transform(batch_data['race'])
    batch_data['gender'] = gender_encoder.transform(batch_data['gender'])
    batch_data['native-country'] = native_country_encoder.transform(batch_data['native-country'])

    # Predict and show results
    preds = model.predict(batch_data)
    label_map = {0: "≤50K", 1: ">50K"}
    batch_data['PredictedClass'] = [label_map[p] for p in preds]

    st.write("✅ Predictions:")
    st.write(batch_data.head())

    # Download predictions
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
