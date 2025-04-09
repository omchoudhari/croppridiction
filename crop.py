import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("crop_yield.csv")
    return df

df = load_data()

# Encode categorical variables
le_crop = LabelEncoder()
df['Crop_Label'] = le_crop.fit_transform(df['Crop'])

le_state = LabelEncoder()
df['State_Label'] = le_state.fit_transform(df['State'])

le_season = LabelEncoder()
df['Season_Label'] = le_season.fit_transform(df['Season'])

# Features and target
features = ['Season_Label', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'State_Label']
target = 'Crop_Label'

X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🌾 Crop Recommendation System")
st.markdown("Enter the conditions below to get a crop recommendation:")

# Input fields
state_input = st.selectbox("Select State", sorted(df['State'].unique()))
season_input = st.selectbox("Select Season", sorted(df['Season'].unique()))
rainfall_input = st.number_input("Annual Rainfall (in mm)", min_value=0.0, step=0.1)
fertilizer_input = st.number_input("Fertilizer Used (in kg/hectare)", min_value=0.0, step=0.1)
pesticide_input = st.number_input("Pesticide Used (in kg/hectare)", min_value=0.0, step=0.1)

if st.button("Recommend Crop"):
    # Encode inputs
    state_enc = le_state.transform([state_input])[0]
    season_enc = le_season.transform([season_input])[0]

    input_data = np.array([[season_enc, rainfall_input, fertilizer_input, pesticide_input, state_enc]])
    prediction = model.predict(input_data)
    predicted_crop = le_crop.inverse_transform(prediction)[0]

    st.success(f"✅ Recommended Crop: **{predicted_crop}**")
