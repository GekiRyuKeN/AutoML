import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.pipeline import make_pipeline
import joblib
import io
from tqdm import tqdm

# Styling to center the image and title in sidebar
st.markdown(
    """
    <style>
        [data-testid=stSidebar]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

# Sidebar for the app
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2172/2172891.png", width=100)
    st.title("Robust AutoML App")
    choice = st.radio("", ["Upload Data", "Basic EDA", "Train Model", "Download Model"])
    st.info("Upload your dataset to build a basic machine learning pipeline!")

# Function to check whether the dataset has been uploaded or not
def is_data_uploaded():
    return os.path.exists("uploaded_data.csv")

# Function to check whether the model has been trained or not  
def is_model_trained():
    return os.path.exists("best_model.pkl")

# For upload option
if choice == "Upload Data":
    if is_data_uploaded():
        os.remove("uploaded_data.csv")
    if is_model_trained():
        os.remove("best_model.pkl")
    
    file = st.file_uploader("Upload Data (CSV only)")
    if file:
        try:
            df = pd.read_csv(file, index_col=None)
            df.to_csv("uploaded_data.csv", index=None)
            st.dataframe(df.head())
            st.success(f"Dataset with {df.shape[0]} rows and {df.shape[1]} columns uploaded successfully!")
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")

# For Basic EDA option
if choice == "Basic EDA":
    if is_data_uploaded():
        st.title("Basic Exploratory Data Analysis")
        try:
            df = pd.read_csv("uploaded_data.csv")
            
            st.write("Dataset Info:")
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            
            st.write("Dataset Description:")
            st.write(df.describe())
            
            st.write("Data Types:")
            st.write(df.dtypes)
            
            # Dynamic Data Visualization
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            categorical_columns = df.select_dtypes(include=['object']).columns
            
            # Data Distribution for Numeric Columns
            if len(numeric_columns) > 0:
                st.write("Data Distribution for Numeric Columns:")
                for column in numeric_columns:
                    fig, ax = plt.subplots()
                    sns.histplot(df[column], kde=True, ax=ax)
                    st.pyplot(fig)
            
            # Pair Plot for Numeric Columns
            if len(numeric_columns) > 1:
                st.write("Pair Plot for Numeric Columns:")
                sns.pairplot(df[numeric_columns])
                st.pyplot()
            
            # Bar Plots for Categorical Columns
            if len(categorical_columns) > 0:
                st.write("Bar Plots for Categorical Columns:")
                for column in categorical_columns:
                    fig, ax = plt.subplots()
                    sns.countplot(y=column, data=df, order=df[column].value_counts().index, ax=ax)
                    st.pyplot(fig)
            
            # 3D Scatter Plot
            if len(numeric_columns) >= 3:
                st.write("3D Scatter Plot:")
                fig = px.scatter_3d(df, x=numeric_columns[0], y=numeric_columns[1], z=numeric_columns[2])
                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error during EDA: {str(e)}")
    else:
        st.title("Basic Exploratory Data Analysis")
        st.info("Please upload a dataset first!")

# For training option
if choice == "Train Model":
    if is_data_uploaded():
        st.title("Robust ML Model Trainer")
        try:
            df = pd.read_csv("uploaded_data.csv")
            
            type = st.selectbox("Select the type of Problem", ["Regression", "Classification"])
            target = st.selectbox("Select the Target Variable", df.columns)
            
            if st.button("Start Training"):
                X = df.drop(target, axis=1)
                y = df[target]

                # Join text columns for TF-IDF
                X = X.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
                
                # Text classification pipeline
                pipeline = make_pipeline(
                    TfidfVectorizer(),
                    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                )

                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train model
                pipeline.fit(X_train, y_train)

                # Evaluate model
                y_pred = pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy}")
                
                # Confusion Matrix
                st.write("Confusion Matrix:")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap='Blues', ax=ax)
                st.pyplot(fig)
                
                # Classification Report
                st.write("Classification Report:")
                report = classification_report(y_test, y_pred, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.write(df_report)

                # Save the model
                joblib.dump(pipeline, "best_model.pkl")
                st.success("Model trained and saved successfully!")

        except Exception as e:
            st.error(f"An error occurred during model training: {str(e)}")
            st.write("Please check your data and ensure it's appropriate for the selected problem type.")
    else:
        st.title("Robust ML Model Trainer")
        st.info("Please upload a dataset first!")

# For download option
if choice == "Download Model":
    if is_data_uploaded():
        if is_model_trained():
            st.title("Download Trained Model")
            with open("best_model.pkl", "rb") as file:
                st.download_button("Download Model", file, "trained_model.pkl")
        else:
            st.title("Download Trained Model")
            st.info("Please train a model first!")
    else:
        st.title("Download Trained Model")
        st.info("Please upload a dataset first!")
