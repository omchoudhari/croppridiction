import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import streamlit as st

# Load dataset
file_path = "C:\Users\Admin\Documents\new project\crop_yield.csv"
df = pd.read_csv(file_path )

# Drop unnecessary columns
X = df.drop(columns=['Yield', 'Crop_Year'])  # 'Crop_Year' might not be useful for prediction
y = df['Yield']

# Encode categorical variables
label_encoders = {}
for col in ['Crop', 'Season', 'State']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Standardize numerical features
scaler = StandardScaler()
X[['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']] = scaler.fit_transform(
    X[['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "crop_yield_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

# Streamlit Web App
def main():
    st.title("Crop Yield Prediction")
    
    crop = st.selectbox("Select Crop", df['Crop'].unique())
    season = st.selectbox("Select Season", df['Season'].unique())
    state = st.selectbox("Select State", df['State'].unique())
    area = st.number_input("Enter Area", min_value=0.0, format="%.2f")
    production = st.number_input("Enter Production", min_value=0.0, format="%.2f")
    annual_rainfall = st.number_input("Enter Annual Rainfall", min_value=0.0, format="%.2f")
    fertilizer = st.number_input("Enter Fertilizer Usage", min_value=0.0, format="%.2f")
    pesticide = st.number_input("Enter Pesticide Usage", min_value=0.0, format="%.2f")
    
    if st.button("Predict Yield"):
        model = joblib.load("crop_yield_model.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoders = joblib.load("label_encoders.pkl")
        
        # Encode input values
        crop_encoded = label_encoders['Crop'].transform([crop])[0]
        season_encoded = label_encoders['Season'].transform([season])[0]
        state_encoded = label_encoders['State'].transform([state])[0]
        
        # Scale input values
        input_data = np.array([[crop_encoded, season_encoded, state_encoded, area, production, annual_rainfall, fertilizer, pesticide]])
        input_data_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_data_scaled)[0]
        st.success(f"Predicted Yield: {prediction:.2f}")

if __name__ == "__main__":
    main()
