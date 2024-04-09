import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import streamlit as st

# Load the CSV file into a DataFrame
diabetes_df = pd.read_csv('diabetes.csv')

# Assuming the CSV file contains features (X) and a target variable (y)
X = diabetes_df.drop('Outcome', axis=1)  # Features
y = diabetes_df['Outcome']  # Target variable

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train_ss=ss.fit_transform(x_train)
x_test_ss=ss.transform(x_test)

# Create and fit the SVC model
model= SVC(kernel='sigmoid', C=1, random_state=42)
model.fit(x_train_ss, y_train)

# Save the model using pickle
with open('diabetes_model.sav', 'wb') as file:
    pickle.dump(model, file)

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

    prediction = model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                   Insulin, BMI, DiabetesPedigreeFunction, Age]])
    if prediction == 1:
        st.error('You have diabetes!')
    else:
        st.success('You do not have diabetes.')
