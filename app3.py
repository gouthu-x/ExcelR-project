
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
from sklearn.neighbors import KNeighborsClassifier  # Import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Set page title and layout
st.set_page_config(page_title="Liver Disease Prediction", layout="wide")
st.title("Liver Disease Prediction App")
# About section
st.sidebar.title("About")  # Add a title to the sidebar
st.sidebar.markdown("""
This is a Liver Disease Prediction App built using Streamlit and various machine learning models. 
It aims to predict the likelihood of liver disease based on patient attributes. 
The app provides an interactive interface for users to upload their data, select a model, 
and visualize the results.

**Key Features:**
* Data Input: Upload your dataset (CSV format).
* Model Selection: Choose from different models (Logistic Regression, KNN, SVM, Naive Bayes, Random Forest).
* Prediction and Evaluation: Get predictions and performance metrics.
* Visualization: Explore results with confusion matrix and feature importance plots.

**Disclaimer:** This app is for informational and educational purposes only. 
It should not be used as a substitute for professional medical advice.""")



# Function to load and preprocess data
@st.cache_data  # Cache the data loading and preprocessing for performance
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # Data Preprocessing
    # Replace outliers based on the IQR method
    def replace_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        replace_outliers_iqr(data, col)

    x = data.drop("category", axis=1)  # Features (replace category column)
    y = data["category"]  # Target values

    # Convert non-numeric columns to numeric using one-hot encoding or label encoding
    X = pd.get_dummies(x, drop_first=True)

    # Check for missing or invalid values and handle them
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.mean())  # Replace missing values with column mean

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(uploaded_file)

    # Model selection
    model_choice = st.selectbox("Select a model:",
                                ["Logistic Regression", "KNN", "SVM", "Naive Bayes", "Random Forest"])

    # Train and evaluate the selected model
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_choice == "SVM":
        model = SVC(kernel='linear')
    elif model_choice == "Naive Bayes":
        model = GaussianNB()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Display results
    st.write(f"**Model:** {model_choice}")
    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred))

    # Visualization: Confusion Matrix
    st.write("**Confusion Matrix:**")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Visualization: Feature Importance (for Random Forest)
    if model_choice == "Random Forest":
        st.write("**Feature Importance:**")
        importances = model.feature_importances_
        features = X_train.columns  # Assuming X_train is a DataFrame

        # Create a DataFrame for better visualization
        feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots()
        ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        st.pyplot(fig)