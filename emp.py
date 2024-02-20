import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define the Streamlit app
def main():
    st.title('Employee Attrition Prediction')
    st.write("Aim: Develop a model to predict the likelihood of employee attrition in a company.")
    st.write("Description: Utilize HR data to build a classification model that predicts whether an employee is likely to leave the company.")
    st.write("Technologies: Python, Pandas, Scikit-learn.")
    st.write("What You Learn: Advanced classification techniques, feature engineering for HR analytics.")

    # Load the dataset
    emp = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

    # Dataset Preview section
    dataset_preview(emp)

    # Exploratory Data Analysis (EDA) section
    eda_section(emp)

    # Preprocess the data
    preprocessed_data = preprocess_data(emp)

    # Train the logistic regression model
    model = train_model(preprocessed_data)

    # Evaluate the model
    accuracy = evaluate_model(model, preprocessed_data)

    # Display the accuracy
    model_accuracy(accuracy)

# Function to display Dataset Preview
def dataset_preview(emp):
    st.header('Dataset Preview')
    st.write(emp.head())

# Function to perform Exploratory Data Analysis (EDA)
def eda_section(emp):
    st.header('Exploratory Data Analysis (EDA)')

    # Attrition Distribution
    attrition_distribution(emp)

    # Attrition by Age
    attrition_by_age(emp)

# Function to display Attrition Distribution
def attrition_distribution(emp):
    st.subheader('Attrition Distribution')
    attrition_count = emp['Attrition'].value_counts()
    st.bar_chart(attrition_count)

# Function to display Attrition by Age
def attrition_by_age(emp):
    st.subheader('Attrition by Age')
    fig, ax = plt.subplots(figsize=(10,6))
    sns.histplot(x='Age', hue='Attrition', data=emp, bins=30, kde=True, palette='Set2', ax=ax)
    ax.set_title('Attrition by Age', fontsize=18)
    ax.set_xlabel('Age', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.legend(title='Attrition', labels=['No', 'Yes'])
    st.pyplot(fig)

# Function to preprocess the data
def preprocess_data(emp):
    # One-hot encode categorical variables
    emp_encoded = pd.get_dummies(emp, drop_first=True)

    # Select numerical columns to scale
    num_cols = emp.select_dtypes(include=['int64', 'float64']).columns

    # Applying standard scaling to numerical columns
    scaler = StandardScaler()
    emp_encoded[num_cols] = scaler.fit_transform(emp_encoded[num_cols])

    return emp_encoded

# Function to train the logistic regression model
def train_model(data):
    # Splitting the data into features and target
    X = data.drop('Attrition_Yes', axis=1)
    y = data['Attrition_Yes']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)

    # Train the logistic regression model
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)

    return logistic_model

# Function to evaluate the model
def evaluate_model(model, data):
    # Splitting the data into features and target
    X = data.drop('Attrition_Yes', axis=1)
    y = data['Attrition_Yes']

    # Making predictions
    y_pred = model.predict(X)

    # Compute accuracy
    accuracy = accuracy_score(y, y_pred)

    return accuracy

# Function to display Model Accuracy
def model_accuracy(accuracy):
    st.header('Model Accuracy')
    st.write(f"The accuracy of the model is: {accuracy}")

if __name__ == '__main__':
    main()
