
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --- Streamlit App ---
st.title("Disease Prediction App")

# 1. Data Upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # 2. EDA (Exploratory Data Analysis)
    st.subheader("Data Exploration")

    # Display basic information about the dataset
    st.write("Data Shape:", data.shape)
    st.write("Data Info:\n", data.info())
    st.write("Data Description:\n", data.describe())

    # Handle missing values (if any)
    data.fillna(0, inplace=True)  # Replace missing values with 0



    # Detect and handle outliers (using IQR method - you can modify this)
    def replace_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        replace_outliers_iqr(data, col)

    # Correlation analysis
    st.subheader("Correlation Matrix")
    numerical_data = data.select_dtypes(include=['number'])
    correlation_matrix = numerical_data.corr()
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='YlGnBu', center=0,
                linewidths=1, linecolor='white', square=True, cbar_kws={"shrink": 0.9}, ax=ax)
    plt.title('Correlation Matrix', fontsize=16, weight='bold')
    plt.xticks(rotation=45, ha='center', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    # 3. Data Preprocessing
    st.subheader("Data Preprocessing")

    # Feature scaling (using StandardScaler) - Apply to numerical features only
    numerical_features = data.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    data_scaled = data.copy()  # Create a copy to avoid modifying the original data
    data_scaled[numerical_features] = scaler.fit_transform(data[numerical_features])

    # One-hot encoding for categorical features (e.g., 'sex')
    data_processed = pd.get_dummies(data_scaled, columns=['sex'], drop_first=True)

    # 4. Model Training (using Logistic Regression)
    st.subheader("Model Training")
    X = data_processed.drop("category", axis=1)  # Features
    y = data_processed["category"]  # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)  # You might need to adjust max_iter
    model.fit(X_train, y_train)

    # Evaluate model performance (optional)
    y_pred = model.predict(X_test)
    st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Classification Report:\n", classification_report(y_test, y_pred))

    # 5. Prediction with User Input
    st.subheader("Disease Prediction")

    # Get user input for features using Streamlit widgets
    age = st.number_input("Enter age:", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Enter sex:", ["Male", "Female"])
    albumin = st.number_input("Enter albumin level:")
    alkaline_phosphatase = st.number_input("Enter alkaline phosphatase level:")
    alanine_aminotransferase = st.number_input("Enter alanine aminotransferase level:")
    aspartate_aminotransferase = st.number_input("Enter aspartate aminotransferase level:")
    bilirubin = st.number_input("Enter bilirubin level:")
    cholinesterase = st.number_input("Enter cholinesterase level:")
    cholesterol = st.number_input("Enter cholesterol level:")
    creatinina = st.number_input("Enter creatinina level:")
    gamma_glutamyl_transferase = st.number_input("Enter gamma glutamyl transferase level:")
    protein = st.number_input("Enter protein level:")

    # Create DataFrame for user input
    user_data = pd.DataFrame({
        'age': [age],
        'sex_m': [1 if sex.lower() == 'male' else 0],  # Use lowercase for comparison and 'sex_m'
        'albumin': [albumin],
        'alkaline_phosphatase': [alkaline_phosphatase],
        'alanine_aminotransferase': [alanine_aminotransferase],
        'aspartate_aminotransferase': [aspartate_aminotransferase],
        'bilirubin': [bilirubin],
        'cholinesterase': [cholinesterase],
        'cholesterol': [cholesterol],
        'creatinina': [creatinina],
        'gamma_glutamyl_transferase': [gamma_glutamyl_transferase],
        'protein': [protein]
    })

    # Scale user input using the same scaler
    numerical_features = ['age', 'albumin', 'alkaline_phosphatase', 'alanine_aminotransferase',
                          'aspartate_aminotransferase', 'bilirubin', 'cholinesterase', 'cholesterol',
                          'creatinina', 'gamma_glutamyl_transferase', 'protein']
    scaled_user_data = scaler.transform(user_data[numerical_features])
    user_data_scaled = pd.DataFrame(scaled_user_data, columns=numerical_features, index=user_data.index)

    # Concatenate and ensure correct order
    user_data_final = pd.concat([user_data_scaled, user_data[['sex_m']]], axis=1)  # Correct order
    user_data_final = user_data_final[['age', 'albumin', 'alkaline_phosphatase', 'alanine_aminotransferase',
                                       'aspartate_aminotransferase', 'bilirubin', 'cholinesterase', 'cholesterol',
                                       'creatinina', 'gamma_glutamyl_transferase', 'protein', 'sex_m']]

    # Make prediction using the model
    if st.button("Predict"):  # Add a button to trigger prediction
        prediction = model.predict(user_data_final)
        st.write("Predicted Disease Category:", prediction[0])