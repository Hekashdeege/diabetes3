import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# URL of the PIMA Diabetes Dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

# Read the dataset directly into a pandas DataFrame
diabetes_dataset = pd.read_csv(url)

# Save the dataset to a CSV file
diabetes_dataset.to_csv("diabetes.csv", index=False)

# Title of the app
st.title('Diabetes Prediction App')

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')

def user_input_features():
    pregnancies = st.sidebar.number_input('Pregnancies', 0, 20, 0)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 20)
    insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 79.0)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 20.0)
    diabetes_pedigree_function = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.42, 0.471)
    age = st.sidebar.slider('Age', 21, 81, 33)
    
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Separating the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Data Standardization
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
Y = diabetes_dataset['Outcome']

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Training the Model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Model Evaluation
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

st.write('Accuracy score of the training data: {:.2f}'.format(training_data_accuracy))
st.write('Accuracy score of the test data: {:.2f}'.format(test_data_accuracy))

# Making a Predictive System
st.subheader('Prediction')
st.write(input_df)

input_data = np.asarray(input_df)
input_data_reshaped = input_data.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)

if prediction[0] == 0:
    st.write('The person is not diabetic')
else:
    st.write('The person is diabetic')

