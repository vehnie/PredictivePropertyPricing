import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import lightgbm as lgb
import joblib

"""
# :house::money_with_wings: Predictive Property Pricing Model for Portugal

The project aims to leverage machine learning techniques to build a predictive model that estimates
property prices across various regions in Portugal. The model will cater to three different scenarios:
property sales, rental rates, and vacation property pricing.
"""
 
st.divider()

data = {'Location': [],
        'Rooms': [],
        'Area': [],
        'Bathrooms':[],
        'Condition':[],
        'AdsType':[],
        'Latitude':[],
        'Longitude':[],
        'Region':[]}

df = pd.DataFrame(data)


def geocode_location(location):
    geolocator = Nominatim(user_agent="location_converter")
    try:
        geo = geolocator.geocode(location)
        if geo is not None:
            return geo.latitude, geo.longitude
        else:
            return None
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        st.error("Geocoding Error: Please enter a valid location.")
        return None
    
def coordinates_to_address(latitude, longitude):
    geolocator = Nominatim(user_agent="coordinate_converter")
    location = geolocator.reverse((latitude, longitude), exactly_one=True)
    if location:
        return location.address
    else:
        return "Address not found"


def coordinates_to_country(latitude, longitude):
    geolocator = Nominatim(user_agent="coordinate_converter")
    location = geolocator.reverse((latitude, longitude), exactly_one=True)
    if location:
        return location.raw.get('address', {}).get('country', '')
    else:
        return None
    

latitude= 0.0
longitude= 0.0
address= None

"""

## :round_pushpin: Where is your property located?

"""
input_location = st.text_input("Enter a location")

if input_location:
    result = geocode_location(input_location)
    if result:

        latitude, longitude = result
        address = coordinates_to_address(latitude, longitude)
        st.success(f"Location Found! Address: {address}")
        df.at[0, 'Latitude'] = latitude
        df.at[0, 'Longitude'] = longitude
        country = coordinates_to_country(latitude, longitude)

df.at[0, 'Location'] = address

st.divider()

"""

## :bed: How many rooms does your property have?

"""

rooms = st.number_input(label='Enter number of rooms', step=1, format='%d')
st.write('You selected:', rooms, 'rooms')
df.at[0, 'Rooms'] = rooms


st.divider()

"""

## :building_construction: How much floor space does your property have?

"""
area = st.number_input(label='Enter the floor space (in m2)', step=0.1)
st.write('Area selected:', area, "m2")
df.at[0, 'Area'] = area

st.divider()

"""
## :bathtub: How many bathrooms does the property have?
"""

bathrooms = st.number_input(label='Enter the number of bathrooms', step=1, format='%d')
st.write('You selected:', bathrooms, 'bathrooms')
df.at[0, 'Bathrooms'] = bathrooms

st.divider()

"""
## :page_with_curl: What is the condition of your property?
"""

condition = st.selectbox(
    'Please select the most appropriate option',
    ('Used', 'Renovated', 'New', 'Under construction', 'To recovery', 'In ruin'))
st.write('The Condition choosen was:', condition)

n_condition= -1

if condition == "Used":
    df.at[0, 'Condition'] = 5
    n_condition= 5
elif condition == "Renovated":
    df.at[0, 'Condition']= 2
    n_condition= 2
elif condition == "New":
    df.at[0, 'Condition']= 1
    n_condition= 1
elif condition == "Under construction":
    df.at[0, 'Condition']= 4
    n_condition= 4
elif condition == "To recovery":
    df.at[0, 'Condition']= 3
    n_condition= 3
else:
    df.at[0, 'Condition']= 0
    n_condition= 0

st.divider()

"""
## :moneybag: What are you looking for?
"""

adstype = st.selectbox(
    'Please select the most appropriate option',
    ('Sell', 'Rent', 'Vacation'))
st.write('The Ad Type choosen was:', adstype)

n_adstype= -1

if adstype == "Rent":
    df.at[0, 'AdsType'] = 0
    n_adstype= 0
elif adstype == "Sell":
    df.at[0, 'AdsType']= 1
    n_adstype= 1
else:
    df.at[0, 'AdsType']= 2
    n_adstype= 2


st.divider()

zone_ranges = {
    '1': {'Latitude': (39.46, np.inf), 'Longitude': (np.NINF, -8.98)},
    '2': {'Latitude': (39.16, 39.46), 'Longitude': (np.NINF, -8.98)},
    '3': {'Latitude': (38.93, 39.16), 'Longitude': (np.NINF, -8.98)},
    '4': {'Latitude': (38.61, 38.93), 'Longitude': (np.NINF, -8.98)},
    '5': {'Latitude': (np.NINF, 38.61), 'Longitude': (np.NINF, -8.98)},
    '6': {'Latitude': (41.77, np.inf), 'Longitude': (-8.98, -8.39)},
    '7': {'Latitude': (41.53, 41.77), 'Longitude': (-8.98, -8.39)},
    '8': {'Latitude': (41.31, 41.53), 'Longitude': (-8.98, -8.39)},
    '9': {'Latitude': (40.99, 41.31), 'Longitude': (-8.98, -8.39)},
    '10': {'Latitude': (40.69, 40.99), 'Longitude': (-8.98, -8.39)},
    '11': {'Latitude': (40.32, 40.69), 'Longitude': (-8.98, -8.39)},
    '12': {'Latitude': (39.89, 40.32), 'Longitude': (-8.98, -8.39)},
    '13': {'Latitude': (39.46, 39.89), 'Longitude': (-8.98, -8.39)},
    '14': {'Latitude': (39.16, 39.46), 'Longitude': (-8.98, -8.39)},
    '15': {'Latitude': (38.93, 39.16), 'Longitude': (-8.98, -8.39)},
    '16': {'Latitude': (38.61, 38.93), 'Longitude': (-8.98, -8.39)},
    '17': {'Latitude': (38.32, 38.61), 'Longitude': (-8.98, -8.39)},
    '18': {'Latitude': (37.87, 38.32), 'Longitude': (-8.98, -8.39)},
    '19': {'Latitude': (37.52, 37.87), 'Longitude': (-8.98, -8.39)},
    '20': {'Latitude': (np.NINF, 37.52), 'Longitude': (-8.98, -8.39)},
    '21': {'Latitude': (41.77, np.inf), 'Longitude': (-8.39, -7.98)},
    '22': {'Latitude': (41.53, 41.77), 'Longitude': (-8.39, -7.98)},
    '23': {'Latitude': (41.31, 41.53), 'Longitude': (-8.39, -7.98)},
    '24': {'Latitude': (40.99, 41.31), 'Longitude': (-8.39, -7.98)},
    '25': {'Latitude': (40.69, 40.99), 'Longitude': (-8.39, -7.98)},
    '26': {'Latitude': (40.32, 40.69), 'Longitude': (-8.39, -7.98)},
    '27': {'Latitude': (39.89, 40.32), 'Longitude': (-8.39, -7.98)},
    '28': {'Latitude': (39.46, 39.89), 'Longitude': (-8.39, -7.98)},
    '29': {'Latitude': (39.16, 39.46), 'Longitude': (-8.39, -7.98)},
    '30': {'Latitude': (38.93, 39.16), 'Longitude': (-8.39, -7.98)},
    '31': {'Latitude': (38.61, 38.93), 'Longitude': (-8.39, -7.98)},
    '32': {'Latitude': (38.32, 38.61), 'Longitude': (-8.39, -7.98)},
    '33': {'Latitude': (37.87, 38.32), 'Longitude': (-8.39, -7.98)},
    '34': {'Latitude': (37.52, 37.87), 'Longitude': (-8.39, -7.98)},
    '35': {'Latitude': (np.NINF, 37.52), 'Longitude': (-8.39, -7.98)},
    '36': {'Latitude': (41.77, np.inf), 'Longitude': (-7.98, -7.54)},
    '37': {'Latitude': (41.53, 41.77), 'Longitude': (-7.98, -7.54)},
    '38': {'Latitude': (41.31, 41.53), 'Longitude': (-7.98, -7.54)},
    '39': {'Latitude': (40.99, 41.31), 'Longitude': (-7.98, -7.54)},
    '40': {'Latitude': (40.69, 40.99), 'Longitude': (-7.98, -7.54)},
    '41': {'Latitude': (40.32, 40.69), 'Longitude': (-7.98, -7.54)},
    '42': {'Latitude': (39.89, 40.32), 'Longitude': (-7.98, -7.54)},
    '43': {'Latitude': (39.46, 39.89), 'Longitude': (-7.98, -7.54)},
    '44': {'Latitude': (39.16, 39.46), 'Longitude': (-7.98, -7.54)},
    '45': {'Latitude': (38.93, 39.16), 'Longitude': (-7.98, -7.54)},
    '46': {'Latitude': (38.61, 38.93), 'Longitude': (-7.98, -7.54)},
    '47': {'Latitude': (38.32, 38.61), 'Longitude': (-7.98, -7.54)},
    '48': {'Latitude': (37.87, 38.32), 'Longitude': (-7.98, -7.54)},
    '49': {'Latitude': (37.52, 37.87), 'Longitude': (-7.98, -7.54)},
    '50': {'Latitude': (np.NINF, 37.52), 'Longitude': (-7.98, -7.54)},
    '51': {'Latitude': (41.77, np.inf), 'Longitude': (-7.54, -7.02)},
    '52': {'Latitude': (41.53, 41.77), 'Longitude': (-7.54, -7.02)},
    '53': {'Latitude': (41.31, 41.53), 'Longitude': (-7.54, -7.02)},
    '54': {'Latitude': (40.99, 41.31), 'Longitude': (-7.54, -7.02)},
    '55': {'Latitude': (40.69, 40.99), 'Longitude': (-7.54, -7.02)},
    '56': {'Latitude': (40.32, 40.69), 'Longitude': (-7.54, -7.02)},
    '57': {'Latitude': (39.89, 40.32), 'Longitude': (-7.54, -7.02)},
    '58': {'Latitude': (39.46, 39.89), 'Longitude': (-7.54, -7.02)},
    '59': {'Latitude': (39.16, 39.46), 'Longitude': (-7.54, -7.02)},
    '60': {'Latitude': (38.93, 39.16), 'Longitude': (-7.54, -7.02)},
    '61': {'Latitude': (38.61, 38.93), 'Longitude': (-7.54, -7.02)},
    '62': {'Latitude': (38.32, 38.61), 'Longitude': (-7.54, -7.02)},
    '63': {'Latitude': (37.87, 38.32), 'Longitude': (-7.54, -7.02)},
    '64': {'Latitude': (37.52, 37.87), 'Longitude': (-7.54, -7.02)},
    '65': {'Latitude': (np.NINF, 37.52), 'Longitude': (-7.54, -7.02)},
    '66': {'Latitude': (41.77, np.inf), 'Longitude': (-7.02, np.inf)},
    '67': {'Latitude': (41.58, 41.77), 'Longitude': (-7.02, np.inf)},
    '68': {'Latitude': (41.31, 41.58), 'Longitude': (-7.02, np.inf)},
    '69': {'Latitude': (40.99, 41.31), 'Longitude': (-7.02, np.inf)},
    '70': {'Latitude': (40.69, 40.99), 'Longitude': (-7.02, np.inf)},
    '71': {'Latitude': (40.32, 40.69), 'Longitude': (-7.02, np.inf)},
    '72': {'Latitude': (39.89, 40.32), 'Longitude': (-7.02, np.inf)},
    '73': {'Latitude': (39.46, 39.89), 'Longitude': (-7.02, np.inf)},
    '74': {'Latitude': (38.93, 39.16), 'Longitude': (-7.02, np.inf)},
    '75': {'Latitude': (np.NINF, 38.93), 'Longitude': (-7.02, np.inf)},
}

def get_region(latitude, longitude):
    for region, ranges in zone_ranges.items():
        lat_range = ranges['Latitude']
        long_range = ranges['Longitude']
        if (lat_range[0] <= latitude <= lat_range[1]) and (long_range[0] <= longitude <= long_range[1]):
            return region
    return None




df= df.dropna()

region = get_region(latitude, longitude)
df.at[0, 'Region']= region

df = df.drop('Latitude', axis=1)
df = df.drop('Longitude', axis=1)
df = df.drop('Location', axis=1)



if st.button("Submit"):
    verificar= False

    if input_location and rooms and area and bathrooms and condition and adstype:
        if area <= 0:
            st.error("Please enter a positive value for the area.")
        if rooms < 0:
            st.error("Please enter a positive value for the rooms.")
        if bathrooms < 0:
            st.error("Please enter a positive value for the bathrooms.")

        if country != 'Portugal':
            st.error("Please enter a location within Portugal.")
        
        if area > 0 and rooms >= 0 and bathrooms >=0 and country == 'Portugal':
            verificar= True


        if verificar:
            
        
            if n_adstype == 0:

                df['Region'] = df['Region'].astype(int)
                df = df.drop('AdsType', axis=1)
                intervalo_preco= None
                loaded_model = joblib.load('model_0.pkl')

                df.at[0, 'Rooms']= rooms
                df.at[0, 'Area']= area
                df.at[0, 'Bathrooms']= bathrooms
                df.at[0, 'Condition']= n_condition

                scaler = joblib.load('scaler_0.pkl')

                new_property_features_scaled = scaler.transform(df)
                probabilities = loaded_model.predict(new_property_features_scaled)
                predicted_intervals = np.argmax(probabilities, axis=1)
                predicted_interval = np.argmax(probabilities)

                df['Price_Range']= predicted_interval

                if predicted_interval== 0:
                    intervalo_preco= "0-500€"
                elif predicted_interval== 1:
                    intervalo_preco= "500-1000€"
                elif predicted_interval== 2:
                    intervalo_preco= "1000-3500€"
                elif predicted_interval==3:
                    intervalo_preco= "3500-7500€"
                elif predicted_interval== 4:
                    intervalo_preco= "7500-25000€"
                else:
                    intervalo_preco= "25000€+"



            elif n_adstype== 1:

                df['Region'] = df['Region'].astype(int)
                df = df.drop('AdsType', axis=1)
                intervalo_preco= None
                loaded_model = joblib.load('model_1.pkl')

                df.at[0, 'Rooms']= rooms
                df.at[0, 'Area']= area
                df.at[0, 'Bathrooms']= bathrooms
                df.at[0, 'Condition']= n_condition

                scaler = joblib.load('scaler_1.pkl')

                new_property_features_scaled = scaler.transform(df)
                probabilities = loaded_model.predict(new_property_features_scaled)
                predicted_intervals = np.argmax(probabilities, axis=1)
                predicted_interval = np.argmax(probabilities)

                df['Price_Range']= predicted_interval

                if predicted_interval== 0:
                    intervalo_preco= "0-25000€"
                elif predicted_interval== 1:
                    intervalo_preco= "25000-75000€"
                elif predicted_interval== 2:
                    intervalo_preco= "75000-150000€"
                elif predicted_interval==3:
                    intervalo_preco= "150000-350000€"
                elif predicted_interval== 4:
                    intervalo_preco= "350000-700000€"
                else:
                    intervalo_preco= "700000€+"



            elif n_adstype== 2:

                df['Region'] = df['Region'].astype(int)
                df = df.drop('AdsType', axis=1)
                intervalo_preco= None
                loaded_model = joblib.load('model_2.pkl')
                scaler = joblib.load('scaler_2.pkl')

                df.at[0, 'Rooms']= rooms
                df.at[0, 'Area']= area
                df.at[0, 'Bathrooms']= bathrooms
                df.at[0, 'Condition']= n_condition

                new_property_features_scaled = scaler.transform(df)
                probabilities = loaded_model.predict(new_property_features_scaled)
                predicted_intervals = np.argmax(probabilities, axis=1)
                predicted_interval = np.argmax(probabilities)

                df['Price_Range']= predicted_interval

                if predicted_interval== 0:
                    intervalo_preco= "0-500€"
                elif predicted_interval== 1:
                    intervalo_preco= "500-2500€"
                elif predicted_interval== 2:
                    intervalo_preco= "2500-5000€"
                elif predicted_interval==3:
                    intervalo_preco= "5000-7500€"
                elif predicted_interval== 4:
                    intervalo_preco= "7500-15000€"
                else:
                    intervalo_preco= "15000€+"



            st.divider()

            st.write("# :bookmark_tabs: Summary of Input Data")
            st.write("### :world_map: Location")
            st.write(f"Address: {coordinates_to_address(latitude, longitude)}")
            st.write(f"Latitude: {latitude}, Longitude: {longitude}")
            
            st.write("### :scroll: Property Details")
            st.write(f":bed: Rooms: {rooms}")
            st.write(f":building_construction: Area: {area} m2")
            st.write(f":bathtub: Bathrooms: {bathrooms}")
            st.write(f":page_with_curl: Condition: {condition}")
            st.write(f":moneybag: Ad Type: {adstype}")

            st.divider()

            st.title(':robot_face::moneybag: Predicted Price Range')
            if n_adstype== 0:
                st.success(f"The expected rental price for your property is : {intervalo_preco} per month")
            elif n_adstype== 1:
                st.success(f"The expected price for selling your property is : {intervalo_preco}")
            elif n_adstype== 2:
                st.success(f"The expected price for renting your property for vacation is : {intervalo_preco}")

            st.divider()

            st.write(":copyright: Project developed by David Volovei & Miguel Venâncio")