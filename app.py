import pickle
import streamlit as st

with open('diabetes.sav', 'rb') as file:
    model = pickle.load(file)

# Page title
st.title('Diabetes Prediction using ML')
# Input fields
col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

with col1:
    Pregnancies = st.number_input('Pregnancies Value', min_value=0, max_value=15, step=1)

with col2:
    Glucose = st.number_input('Glucose Level', min_value=70, max_value=1000, step=1)

with col3:
    BloodPressure = st.number_input('BloodPressure value', min_value=40, max_value=370, step=1)

with col4:
    SkinThickness = st.number_input('SkinThickness Value')

with col5:
    Insulin = st.number_input('Insulin Value')

with col6:
    BMI = st.number_input('BMI Value')

with col7:
    DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction Value')

with col8:
    Age = st.number_input('Age Value')

# Default value for prediction
prediction = ''

# Prediction button
if st.button('Predict Diabetes'):
    # Ensure input data is valid and convert to appropriate types
    try:
        input_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness,
                       Insulin, BMI, DiabetesPedigreeFunction, Age]]
        # Normalize input data if needed (e.g., scaling to match model training data)
        # Make predictions using the loaded model
        prediction = Diabetes_model.predict(input_data)
        if prediction[0] == 1:
            st.error('You have diabetes!')
        else:
            st.success('You do not have diabetes.')
    except ValueError:
        st.error('Please enter valid numerical values for all input fields.')
