import streamlit as st
import pickle
import numpy as np

# Load the trained model from a pickle file
MODEL_FILE = 'phones_rf.pkl'

@st.cache_resource
def load_model():
    with open(MODEL_FILE, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Define the input features
st.title('Mobile Price Prediction')
st.write('Enter the specifications of the mobile phone:')

battery = st.number_input('Battery (mAh)', min_value=0)
processor = st.checkbox('Processor')
ram = st.number_input('RAM (GB)', min_value=0)
storage = st.number_input('Storage (GB)', min_value=0)
rear_camera = st.checkbox('Rear Camera')
front_camera = st.checkbox('Front Camera')
wi_fi = st.checkbox('Wi-Fi')
bluetooth = st.checkbox('Bluetooth')
gps = st.checkbox('GPS')
no_of_sims = st.number_input('Number of SIMs', min_value=1, max_value=4)
three_g = st.checkbox('3G')
lte_4g = st.checkbox('4G LTE')
resolution_y = st.number_input('Resolution Y', min_value=0)
resolution_x = st.number_input('Resolution X', min_value=0)

def predict_price(battery, processor, ram, storage, rear_camera, front_camera,
                  wi_fi, bluetooth, gps, no_of_sims, three_g, lte_4g, resolution_y, resolution_x):
    # Convert boolean features to integers
    processor = int(processor)
    rear_camera = int(rear_camera)
    front_camera = int(front_camera)
    wi_fi = int(wi_fi)
    bluetooth = int(bluetooth)
    gps = int(gps)
    three_g = int(three_g)
    lte_4g = int(lte_4g)

    # Prepare the input data with feature names
    feature_names = ['Battery', 'Processor', 'RAM', 'Storage', 'Rear Camera', 'Front Camera',
                     'Wi-Fi', 'Bluetooth', 'GPS', 'Number of SIMs', '3G', '4G LTE',
                     'Resolution Y', 'Resolution X']
    input_data = np.array([[battery, processor, ram, storage, rear_camera, front_camera,
                            wi_fi, bluetooth, gps, no_of_sims, three_g, lte_4g, resolution_y, resolution_x]])

    # Perform inference
    price_usd = model.predict(input_data)[0]
    
    return price_usd

if st.button('Predict Price'):
    price_usd = predict_price(battery, processor, ram, storage, rear_camera, front_camera,
                              wi_fi, bluetooth, gps, no_of_sims, three_g, lte_4g, resolution_y, resolution_x)
    st.write(f'Predicted Price: {price_usd:.2f} INR')
