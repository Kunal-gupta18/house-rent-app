import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Streamlit Title
st.title("🏠 House Rent Prediction App")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("rent.csv")
    return df

df = load_data()

# Check if 'Location' column exists
if 'Location' not in df.columns:
    st.error("❌ 'Location' column not found in CSV file. Please check rent.csv.")
    st.stop()

# Encode 'Location' column
le = LabelEncoder()
df['Location'] = le.fit_transform(df['Location'])

# Train Model
X = df[['Size', 'BHK', 'Location']]
y = df['Rent']
model = LinearRegression()
model.fit(X, y)

# User Input Section
st.header("📋 Enter House Details:")

size = st.slider("Size (sq. ft.)", min_value=300, max_value=5000, step=50, value=1000)
bhk = st.selectbox("BHK", sorted(df['BHK'].unique()))
location = st.selectbox("Location", le.classes_)

# Predict Rent
if st.button("Predict Rent"):
    location_encoded = le.transform([location])[0]
    input_data = pd.DataFrame([[size, bhk, location_encoded]], columns=['Size', 'BHK', 'Location'])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Rent: ₹{int(prediction):,} per month")
