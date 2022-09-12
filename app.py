import pandas as pd
import numpy as np
import streamlit as st
import pickle as pkl
import warnings as warning
warning.filterwarnings('ignore')


model = pkl.load(open('xgb_c_model.pkl', 'rb'))
scaler = pkl.load(open('scaler.pkl','rb'))

df = pd.read_csv('https://raw.githubusercontent.com/vishalkrishna90/CHENNAI-HOUSE-PRICE-PREDICTION/main/clean_chennai_house.csv')

st.title('Chennai House Price Prediction')

st.image('i.jpg')

int_sqfts,mzzones,no_of_roomss = st.columns(3)
with int_sqfts:
    int_sqft = st.number_input('Enter Area Size (Int Sqft):', 500, 3000)
with mzzones:
    mzzone = st.selectbox('Select Mzzone:', sorted(df['MZZONE'].unique()) )
with no_of_roomss:
    no_of_rooms = st.selectbox('Select No Of Rooms:', sorted(df['N_ROOM'].unique()))

areas,park_facilitys = st.columns(2)  
with areas:
    area = st.selectbox('Select Area:', sorted(df['AREA'].unique()))
with park_facilitys:
    park_facility = st.selectbox('Select Park Facility:', sorted(df['PARK_FACIL'].unique()))

streets,build_types = st.columns(2)
with streets:
    street = st.selectbox('Select Street Type:', sorted(df['STREET'].unique()))
with build_types:
    build_type = st.selectbox('Select Build Type:', sorted(df['BUILDTYPE'].unique()))

result = st.button('Result')

mz = 1
if mzzone == 'A':
    mz = 1
elif mzzone == 'C':
    mz = 2
elif mzzone == 'I':
    mz = 3
elif mzzone == 'RH':
    mz = 4
elif mzzone == 'RL':
    mz = 5
else:
    mz = 6



pr = 1
if park_facility == 'No':
    pr = 1
elif park_facility == 'Yes':
    pr = 2
    

stt = 1
if street == 'No Access':
    stt = 1
elif street == 'Paved':
    stt = 2
else:
    stt = 3


ar = 1
if area == 'Karapakkam':
    ar = 1
elif area == 'Adyar':
    ar = 2
elif area == 'Chrompet':
    ar = 3
elif area == 'Velachery':
    ar = 4
elif area == 'KK Nagar':
    ar = 5
elif area == 'Anna Nagar':
    ar = 6
elif area == 'T Nagar':
    ar = 7

hous = 0
othrs = 0
if build_type == 'House':
    hous = 1
elif build_type == 'Others':
    othrs = 1
else:
    hous = 0
    othrs = 0


if result:

    x = np.array([[int_sqft,mz,no_of_rooms,ar,pr,stt,hous,othrs]])
    x = scaler.transform(x)
    
    result = model.predict(x)
    st.subheader(f'Predicted House Price Is {round(result[0]/100000,2)} Lac')
    
