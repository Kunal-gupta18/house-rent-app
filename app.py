import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="House Rent Prediction", layout="wide")
st.title("🏠 House Rent Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    st.subheader("📈 Visualizations")

    # Plot Size vs Rent if present
    if 'Size' in df.columns and 'Rent' in df.columns:
        fig1, ax1 = plt.subplots()
        sns.scatterplot(x='Size', y='Rent', data=df, ax=ax1)
        ax1.set_title("Size vs Rent")
        st.pyplot(fig1)

    # Plot City distribution
    if 'City' in df.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.countplot(data=df, x='City', order=df['City'].value_counts().index[:10], ax=ax2)
        ax2.set_title("Top Cities by Listings")
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)

    # ML Model
    st.subheader("🤖 Build Rent Prediction Model")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) >= 2:
        x_feature = st.selectbox("Select Feature (X)", numeric_cols)
        y_target = st.selectbox("Select Target (Y)", [col for col in numeric_cols if col != x_feature])

        X = df[[x_feature]]
        y = df[y_target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success(f"✅ Model Trained! MSE: {mean_squared_error(y_test, y_pred):.2f}")

        input_val = st.number_input(f"Enter {x_feature} to predict {y_target}")
        if input_val:
            prediction = model.predict(np.array([[input_val]]))[0]
            st.info(f"📌 Predicted {y_target}: ₹{prediction:.2f}")
    else:
        st.warning("❗ Dataset should have at least 2 numeric columns.")
else:
    st.info("Upload a CSV file to get started.")
