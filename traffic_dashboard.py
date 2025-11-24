import streamlit as st
import pandas as pd
import numpy as np
import datetime
import sqlite3
import subprocess
import sys
import importlib

# Ensure openpyxl is installed
import openpyxl


# Import ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(style='whitegrid')

# ---- Utility functions ----

def load_data(file):
    return pd.read_excel(file, engine='openpyxl')

def save_to_database(df, db_name="traffic_accidents.db", table_name="accidents"):
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

def load_from_database(db_name="traffic_accidents.db", table_name="accidents"):
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def process_datetime_columns(df):
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except:
            continue
    return df

def handle_missing_values(df):
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    obj_cols = df.select_dtypes(include='object').columns
    df[obj_cols] = df[obj_cols].fillna("Missing")
    return df

def remove_duplicates(df):
    return df.drop_duplicates()

def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def encode_categorical(df):
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders

def prepare_data(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def plot_confusion_matrix(model, X_test, y_test, model_name):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues')
    st.pyplot(fig)

def plot_accuracy_comparison(results):
    models, accuracies = zip(*results)
    fig, ax = plt.subplots()
    sns.barplot(x=list(models), y=list(accuracies), palette="Blues_d", ax=ax)
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=15)
    st.pyplot(fig)

def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC()
    }
    results = []
    best_model = None
    best_acc = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results.append((name, acc))
        st.write(f"**{name}** Accuracy: {acc:.4f}")
        st.write(classification_report(y_test, preds, zero_division=0))
        plot_confusion_matrix(model, X_test, y_test, name)

        if acc > best_acc:
            best_acc = acc
            best_model = model

    plot_accuracy_comparison(results)

    # Feature importance if Random Forest
    if isinstance(best_model, RandomForestClassifier):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(x=[feature_names[i] for i in indices], y=importances[indices], ax=ax)
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
        ax.set_ylabel('Importance')
        ax.set_title('Feature Importances (Random Forest)')
        st.pyplot(fig)

# ---- Streamlit App ----

st.set_page_config(page_title="Traffic Accidents Dashboard", layout="wide")
st.title("🚦 Traffic Accidents Analysis & Prediction Dashboard")

# Sidebar for options
st.sidebar.header("Upload Data & Settings")
uploaded_file = st.sidebar.file_uploader("Upload Excel Dataset", type=["xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.success("Data Loaded Successfully!")

    # Show raw data
    with st.expander("Preview Raw Data"):
        st.dataframe(df.head())

    # Data cleaning options
    st.sidebar.subheader("Data Cleaning")
    clean_data = st.sidebar.button("Clean Data")
    if clean_data:
        df = process_datetime_columns(df)
        df = handle_missing_values(df)
        df = remove_duplicates(df)
        numerical_cols = df.select_dtypes(include='number').columns
        df = remove_outliers(df, numerical_cols)
        st.success("Data cleaned!")

    # Save cleaned data to database
    if st.sidebar.button("Save Data to Database"):
        save_to_database(df)
        st.info("Data saved to database.")

    # Load data from database
    if st.sidebar.button("Load Data from Database"):
        df = load_from_database()
        st.success("Data loaded from database.")
        with st.expander("Preview Loaded Data"):
            st.dataframe(df.head())

    # Show visualizations
    st.header("Exploratory Data Analysis")
    if st.checkbox("Show Visualizations"):
        # Handle categorical encoding for visualization
        df_vis = df.copy()
        cat_cols = df_vis.select_dtypes(include='object').columns
        for col in cat_cols:
            df_vis[col] = df_vis[col].astype(str)

        # Plot visualizations
        with st.spinner("Generating Visualizations..."):
            # 1. Accident Severity Distribution
            st.subheader("Accident Severity Distribution")
            fig, ax = plt.subplots()
            sns.countplot(data=df_vis, x='Accident_severity', palette='Set2', ax=ax)
            st.pyplot(fig)

            # 2. Accidents by Day of Week
            st.subheader("Accidents by Day of Week")
            day_order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
            fig, ax = plt.subplots()
            sns.countplot(data=df_vis, x='Day_of_week', order=day_order, palette='pastel', ax=ax)
            st.pyplot(fig)

            # 3. Severity by Weather Conditions
            st.subheader("Accident Severity by Weather Conditions")
            fig, ax = plt.subplots()
            sns.countplot(data=df_vis, x='Weather_conditions', hue='Accident_severity', palette='coolwarm', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # 4. Severity by Light Conditions
            st.subheader("Accident Severity by Light Conditions")
            fig, ax = plt.subplots()
            sns.countplot(data=df_vis, x='Light_conditions', hue='Accident_severity', palette='flare', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # 5. Correlation Heatmap
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10,8))
            corr = df_vis.select_dtypes(include=np.number).corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)

            # 6. Road Surface vs Light Conditions Heatmap
            st.subheader("Road Surface vs Light Conditions")
            pivot = pd.crosstab(df_vis['Road_surface_conditions'], df_vis['Light_conditions'])
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(pivot, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
            st.pyplot(fig)

            # 7. Top Vehicle Types
            st.subheader("Top 10 Vehicle Types Involved in Accidents")
            top_vehicles = df_vis['Type_of_vehicle'].value_counts().head(10)
            fig, ax = plt.subplots()
            sns.barplot(x=top_vehicles.index, y=top_vehicles.values, palette='rocket', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # 8. Pairplot of key variables
            st.subheader("Relationships Among Key Features")
            selected_cols = ['Number_of_casualties', 'Number_of_vehicles_involved', 'Accident_severity']
            if all(col in df_vis.columns for col in selected_cols):
                sns.pairplot(df_vis[selected_cols], hue='Accident_severity', palette='husl')
                st.pyplot()

            # 9. Violin Plot for Casualties
            st.subheader("Casualties Distribution by Accident Severity")
            if 'Number_of_casualties' in df_vis.columns:
                fig, ax = plt.subplots()
                sns.violinplot(data=df_vis, x='Accident_severity', y='Number_of_casualties', palette='muted', ax=ax)
                st.pyplot(fig)

            # 10. Average Casualties by Road Surface
            st.subheader("Average Casualties by Road Surface")
            if 'Road_surface_conditions' in df_vis.columns and 'Number_of_casualties' in df_vis.columns:
                fig, ax = plt.subplots()
                sns.barplot(
                    data=df_vis, 
                    x='Road_surface_conditions', 
                    y='Number_of_casualties', 
                    estimator=np.mean, 
                    palette='cubehelix',
                    ax=ax
                )
                plt.xticks(rotation=45)
                st.pyplot(fig)

            # 11. Accidents by Vehicle Type
            st.subheader("Accidents by Vehicle Type")
            if 'Type_of_vehicle' in df_vis.columns:
                order = df_vis['Type_of_vehicle'].value_counts().index
                fig, ax = plt.subplots()
                sns.countplot(x='Type_of_vehicle', data=df_vis, order=order, palette='Set1', ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

            # 12. Casualties by Vehicle Type
            st.subheader("Casualties by Vehicle Type")
            if 'Type_of_vehicle' in df_vis.columns and 'Number_of_casualties' in df_vis.columns:
                fig, ax = plt.subplots()
                sns.boxplot(data=df_vis, x='Type_of_vehicle', y='Number_of_casualties', palette='Spectral', ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

            # 13. Line Plot: Casualties over days
            if 'Day_of_week' in df_vis.columns and 'Number_of_casualties' in df_vis.columns:
                casualties_by_day = df_vis.groupby('Day_of_week')['Number_of_casualties'].sum().reindex(day_order)
                fig, ax = plt.subplots()
                casualties_by_day.plot(kind='line', marker='o', color='green', ax=ax)
                ax.set_title("Total Casualties by Day of the Week")
                ax.set_xlabel("Day of Week")
                ax.set_ylabel("Number of Casualties")
                st.pyplot(fig)

            # 14. Area Plot: Vehicles involved over days
            if 'Day_of_week' in df_vis.columns and 'Number_of_vehicles_involved' in df_vis.columns:
                vehicles_by_day = df_vis.groupby('Day_of_week')['Number_of_vehicles_involved'].sum().reindex(day_order)
                fig, ax = plt.subplots()
                vehicles_by_day.plot(kind='area', color='skyblue', alpha=0.7, ax=ax)
                ax.set_title("Total Vehicles Involved by Day of the Week")
                ax.set_xlabel("Day of Week")
                ax.set_ylabel("Number of Vehicles")
                st.pyplot(fig)

            # 15. Pie Chart of Severity
            if 'Accident_severity' in df_vis.columns:
                severity_counts = df_vis['Accident_severity'].value_counts()
                fig, ax = plt.subplots()
                ax.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2'))
                ax.set_title("Accident Severity Distribution")
                st.pyplot(fig)

    # Model Training & Evaluation
    st.header("Model Training & Evaluation")
    target_col = 'Accident_severity'
    if target_col in df.columns:
        # Prepare data
        df_model = df.copy()
        # Encode categorical variables
        for col in df_model.select_dtypes(include='object').columns:
            df_model[col] = df_model[col].astype(str)
        df_model, encoders = encode_categorical(df_model)

        if df_model.isnull().sum().sum() == 0 and target_col in df_model.columns:
            X, y, scaler = prepare_data(df_model, target_col)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            feature_names = df_model.drop(columns=[target_col]).columns

            if st.button("Train and Evaluate Models"):
                train_and_evaluate(X_train, X_test, y_train, y_test, list(feature_names))
        else:
            st.warning("Data contains missing values or target column missing after processing.")
    else:
        st.warning(f"Column '{target_col}' not found in dataset.")

else:
    st.info("Please upload an Excel dataset to begin.")
