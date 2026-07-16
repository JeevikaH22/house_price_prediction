import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

@st.cache_resource
def load_assets():
    with open('house_price_pred.pkl', 'rb') as f:
        model = pickle.load(f)
        return model
    with open('scaler.pkl', 'rb') as fh:
        scaler = pickle.load(fh)
        return scaler
    

@st.cache_resource
def get_fitted_scaler():
    try:
        # Load your raw dataset from the same directory
        df_train = pd.read_csv('Housing_price_data.csv') 
        scaler = StandardScaler()
        scaler.fit(df_train[['area']])
        return scaler
    except FileNotFoundError:
        st.error("Training data file not found! Please place your dataset file in this directory.")
        st.stop()

model = load_assets()
scaler = get_fitted_scaler()

st.title("House Price Prediction App")
st.write("Enter the details below to estimate the market price of the property.")

st.header("Property Dimensions")
raw_area = st.number_input("Total Area (in sq ft)", min_value=100, max_value=20000, value=3000, step=50)

st.header("Rooms and Layout")
col1, col2, col3, col4 = st.columns(4)
with col1:
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
with col2:
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=2)
with col3:
    stories = st.number_input("Stories/Floors", min_value=1, max_value=5, value=2)
with col4:
    parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)

st.header("Features and Amenities")
col5, col6, col7 = st.columns(3)
with col5:
    mainroad = st.selectbox("Mainroad Access", ["yes", "no"])
    guestroom = st.selectbox("Has Guestroom", ["no", "yes"])
with col6:
    basement = st.selectbox("Has Basement", ["no", "yes"])
    hot_water_heating = st.selectbox("Water Heating", ["no", "yes"])
with col7:
    air_conditioning = st.selectbox("Air Conditioning", ["no", "yes"])
    prefered_area = st.selectbox("Preferred Neighborhood", ["no", "yes"])

st.header("Furnishing Status")
furnished_status = st.radio("Select Status", ["furnished", "semi-furnished", "unfurnished"], horizontal=True)

mapping = {"no": 0, "yes": 1}
furnishing_mapping = {"furnished": 0, "semi-furnished": 1, "unfurnished": 2}

scaled_area = scaler.transform([[raw_area]])[0][0]

# Structure matching your renamed model training layout exactly
input_data = pd.DataFrame([{
    'area': scaled_area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'mainroad': mapping[mainroad],
    'guestroom': mapping[guestroom],
    'basement': mapping[basement],
    'hot_water_heating': mapping[hot_water_heating],
    'air_conditioning': mapping[air_conditioning],
    'parking': parking,
    'prefered_area': mapping[prefered_area],
    'furnished': furnishing_mapping[furnished_status]
}])

st.markdown("---")
if st.button("Predict Price", type="primary"):
    log_prediction = model.predict(input_data)[0]
    real_prediction = np.expm1(log_prediction)

    st.success(f"Estimated Value: INR {int(real_prediction):,}")