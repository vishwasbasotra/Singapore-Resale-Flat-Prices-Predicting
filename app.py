import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from geopy.distance import geodesic
import statistics
import numpy as np
import pickle
import re

# -------------------------Reading the data on Lat and Long of all the MRT Stations in Singapore------------------------
data = pd.read_csv('mrt.csv')
mrt_location = pd.DataFrame(data)

# -------------------------------This is the configuration page for our Streamlit Application---------------------------
st.set_page_config(
    page_title="Singapore Resale Flat Prices Prediction",
    page_icon="🏨",
    layout="wide"
)

# -------------------------------This is the sidebar in a Streamlit application, helps in navigation--------------------
with st.sidebar:
    selected = option_menu("Main Menu", ["About Project", "Predictions"],
                           icons=["house", "gear"],
                           styles={"nav-link": {"font": "sans serif", "font-size": "20px", "text-align": "centre"},
                                   "nav-link-selected": {"font": "sans serif", "background-color": "#0072b1"},
                                   "icon": {"font-size": "20px"}
                                   }
                           )

# -----------------------------------------------About Project Section--------------------------------------------------
if selected == "About Project":
    st.markdown("# :blue[Singapore Resale Flat Prices Prediction]")
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown("### :blue[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building, Data Wrangling, "
                "Model Deployment")
    st.markdown("### :blue[Overview :] This project aims to construct a machine learning model and implement "
                "it as a user-friendly online application in order to provide accurate predictions about the "
                "resale values of apartments in Singapore. This prediction model will be based on past transactions "
                "involving resale flats, and its goal is to aid both future buyers and sellers in evaluating the "
                "worth of a flat after it has been previously resold. Resale prices are influenced by a wide variety "
                "of criteria, including location, the kind of apartment, the total square footage, and the length "
                "of the lease. The provision of customers with an expected resale price based on these criteria is "
                "one of the ways in which a predictive model may assist in the overcoming of these obstacles.")
    st.markdown("### :blue[Domain :] Real Estate")

# ------------------------------------------------Predictions Section---------------------------------------------------
if selected == "Predictions":
    st.markdown("# :blue[Predicting Results based on Trained Models]")
    st.markdown("### :orange[Predicting Resale Price (Regression Task) (Accuracy: 87%)]")

    try:
        with st.form("form1"):
            streets = ['ANG MO KIO AVE 4', 'ANG MO KIO AVE 10', 'ANG MO KIO AVE 5',
                'ANG MO KIO AVE 8', 'ANG MO KIO AVE 1', 'ANG MO KIO AVE 3',
                'ANG MO KIO AVE 6', 'ANG MO KIO ST 52', 'ANG MO KIO ST 21',
                'ANG MO KIO ST 31', 'BEDOK RESERVOIR RD', 'BEDOK STH RD',
                'BEDOK NTH ST 3', 'BEDOK NTH AVE 1', 'BEDOK NTH RD',
                'NEW UPP CHANGI RD', 'CHAI CHEE ST', 'BEDOK NTH ST 1',
                'BEDOK NTH AVE 4', 'BEDOK NTH ST 2', 'CHAI CHEE AVE',
                'BEDOK NTH AVE 3', 'BEDOK STH AVE 1', 'BEDOK CTRL',
                'BEDOK NTH AVE 2', 'BEDOK STH AVE 2', 'BEDOK RESERVOIR VIEW',
                'CHAI CHEE RD', 'JLN TENAGA', 'BEDOK STH AVE 3', 'LENGKONG TIGA',
                'SHUNFU RD', 'BISHAN ST 24', 'BISHAN ST 12', 'BISHAN ST 22',
                'BISHAN ST 13', 'BISHAN ST 23', 'BRIGHT HILL DR', 'SIN MING AVE',
                'BT BATOK ST 52', 'BT BATOK WEST AVE 4', 'BT BATOK WEST AVE 2',
                'BT BATOK EAST AVE 4', 'BT BATOK EAST AVE 3', 'BT BATOK ST 21',
                'BT BATOK EAST AVE 5', 'BT BATOK WEST AVE 8', 'BT BATOK ST 11',
                'BT BATOK WEST AVE 6', 'BT BATOK ST 51', 'BT BATOK ST 32',
                'BT BATOK ST 33', 'BT BATOK ST 31', 'BT BATOK CTRL',
                'BT BATOK ST 24', 'BT BATOK ST 25', 'BT BATOK WEST AVE 5',
                'BT BATOK ST 34', 'JLN KLINIK', 'LOWER DELTA RD', 'BT MERAH VIEW',
                'JLN BT HO SWEE', 'JLN BT MERAH', 'TELOK BLANGAH CRES',
                'TELOK BLANGAH HTS', 'KIM TIAN RD', 'BEO CRES', 'CANTONMENT CL',
                'TELOK BLANGAH DR', 'JLN MEMBINA', 'LIM LIAK ST', 'SENG POH RD',
                'LENGKOK BAHRU', 'DEPOT RD', 'KIM TIAN PL', 'REDHILL CL',
                'BOON TIONG RD', 'HOY FATT RD', 'HAVELOCK RD', 'REDHILL LANE',
                'REDHILL RD', 'GANGSA RD', 'PETIR RD', 'BANGKIT RD', 'SAUJANA RD',
                'BT PANJANG RING RD', 'SEGAR RD', 'PENDING RD', 'FAJAR RD',
                'JELAPANG RD', 'SENJA RD', 'SENJA LINK', 'JELEBU RD', 'CASHEW RD',
                'TOH YI DR', 'FARRER RD', 'UPP CROSS ST', 'TG PAGAR PLAZA',
                'CHIN SWEE RD', 'CANTONMENT RD', 'TECK WHYE AVE', 'TECK WHYE LANE',
                'CHOA CHU KANG AVE 4', 'CHOA CHU KANG CTRL', 'CHOA CHU KANG CRES',
                'CHOA CHU KANG AVE 2', 'CHOA CHU KANG AVE 3', 'CHOA CHU KANG DR',
                'CHOA CHU KANG AVE 5', 'JLN TECK WHYE', 'CHOA CHU KANG AVE 1',
                'CHOA CHU KANG ST 62', 'CHOA CHU KANG NTH 6',
                'CHOA CHU KANG ST 64', 'CHOA CHU KANG NTH 5',
                'CHOA CHU KANG ST 51', 'WEST COAST RD', 'CLEMENTI AVE 5',
                'CLEMENTI AVE 2', 'CLEMENTI WEST ST 1', 'CLEMENTI AVE 3',
                'CLEMENTI AVE 4', 'CLEMENTI WEST ST 2', 'WEST COAST DR',
                'CIRCUIT RD', 'ALJUNIED CRES', 'MACPHERSON LANE', 'BALAM RD',
                'PAYA LEBAR WAY', 'EUNOS CRES', 'HAIG RD', 'GEYLANG EAST AVE 1',
                'SIMS DR', 'CASSIA CRES', 'UBI AVE 1', 'ALJUNIED RD', 'PINE CL',
                'JLN TIGA', 'HOUGANG AVE 3', 'HOUGANG AVE 6', 'HOUGANG AVE 5',
                'HOUGANG AVE 1', 'HOUGANG AVE 7', 'HOUGANG ST 22', 'HOUGANG AVE 8',
                'HOUGANG AVE 10', 'HOUGANG ST 11', 'LOR AH SOO', 'HOUGANG ST 92',
                'HOUGANG ST 61', 'HOUGANG AVE 4', 'HOUGANG ST 91', 'HOUGANG ST 51',
                'HOUGANG ST 52', 'HOUGANG AVE 9', 'HOUGANG ST 21', 'HOUGANG CTRL',
                'HOUGANG AVE 2', 'TEBAN GDNS RD', 'JURONG EAST ST 24',
                'JURONG EAST ST 21', 'JURONG EAST ST 13', 'JURONG EAST ST 32',
                'JURONG EAST ST 31', 'TOH GUAN RD', 'PANDAN GDNS', 'BOON LAY AVE',
                'BOON LAY PL', 'JURONG WEST ST 41', 'HO CHING RD',
                'JURONG WEST ST 51', 'JURONG WEST AVE 1', 'JURONG WEST ST 91',
                'KANG CHING RD', 'TAH CHING RD', 'JURONG WEST ST 25',
                'JURONG WEST AVE 3', 'JURONG WEST ST 81', 'JURONG WEST ST 42',
                'JURONG WEST ST 73', 'JURONG WEST ST 61', 'JURONG WEST ST 71',
                'JURONG WEST AVE 5', 'JURONG WEST ST 65', 'YUNG LOH RD',
                'JURONG WEST ST 74', 'JURONG WEST ST 64', 'JURONG WEST CTRL 1',
                'JURONG WEST ST 52', 'JURONG WEST ST 92', 'YUNG SHENG RD',
                'BOON LAY DR', 'JURONG WEST ST 75', 'CORPORATION DR',
                'JURONG WEST ST 93', 'YUNG AN RD', 'JLN BAHAGIA', 'GEYLANG BAHRU',
                'JLN BATU', 'KALLANG BAHRU', 'UPP BOON KENG RD', 'RACE COURSE RD',
                'GLOUCESTER RD', 'BEACH RD', 'BENDEMEER RD', 'WHAMPOA STH',
                'WHAMPOA DR', "ST. GEORGE'S RD", 'JLN DUSUN', 'JLN TENTERAM',
                'BOON KENG RD', 'FARRER PK RD', 'MCNAIR RD', 'AH HOOD RD',
                'WHAMPOA RD', 'CRAWFORD LANE', 'JLN RAJAH', 'MARINE TER',
                'MARINE DR', 'MARINE CRES', 'CHANGI VILLAGE RD', 'PASIR RIS ST 21',
                'PASIR RIS DR 3', 'PASIR RIS DR 6', 'PASIR RIS DR 10',
                'PASIR RIS ST 11', 'PASIR RIS ST 71', 'PASIR RIS DR 1',
                'PASIR RIS DR 4', 'PASIR RIS ST 52', 'PASIR RIS ST 51',
                'PASIR RIS ST 53', 'PASIR RIS ST 12', 'ELIAS RD',
                'PASIR RIS ST 72', 'EDGEDALE PLAINS', 'PUNGGOL FIELD',
                'PUNGGOL CTRL', 'PUNGGOL DR', 'PUNGGOL PL', 'EDGEFIELD PLAINS',
                'STIRLING RD', "C'WEALTH CL", "C'WEALTH CRES", "C'WEALTH DR",
                'DOVER CRES', 'GHIM MOH RD', 'HOLLAND AVE', 'HOLLAND DR',
                'DOVER RD', 'MEI LING ST', 'STRATHMORE AVE', 'QUEENSWAY',
                'HOLLAND CL', 'TANGLIN HALT RD', 'ADMIRALTY LINK', 'SEMBAWANG DR',
                'MONTREAL DR', 'ADMIRALTY DR', 'WELLINGTON CIRCLE', 'SEMBAWANG CL',
                'SEMBAWANG VISTA', 'CANBERRA RD', 'ANCHORVALE LINK',
                'COMPASSVALE LANE', 'RIVERVALE CRES', 'RIVERVALE ST',
                'COMPASSVALE CRES', 'RIVERVALE DR', 'COMPASSVALE WALK',
                'ANCHORVALE DR', 'SENGKANG EAST RD', 'FERNVALE LINK',
                'COMPASSVALE DR', 'RIVERVALE WALK', 'FERNVALE RD',
                'COMPASSVALE RD', 'SENGKANG EAST WAY', 'SENGKANG WEST AVE',
                'COMPASSVALE BOW', 'SENGKANG CTRL', 'COMPASSVALE LINK',
                'COMPASSVALE ST', 'SERANGOON NTH AVE 1', 'SERANGOON AVE 4',
                'SERANGOON CTRL', 'SERANGOON CTRL DR', 'SERANGOON AVE 2',
                'SERANGOON NTH AVE 4', 'SERANGOON NTH AVE 3', 'SERANGOON AVE 3',
                'SERANGOON AVE 1', 'TAMPINES ST 43', 'TAMPINES ST 22',
                'TAMPINES ST 81', 'TAMPINES ST 83', 'TAMPINES ST 44',
                'TAMPINES ST 42', 'TAMPINES ST 41', 'TAMPINES AVE 4',
                'TAMPINES ST 11', 'TAMPINES ST 21', 'TAMPINES ST 23',
                'TAMPINES ST 12', 'SIMEI ST 1', 'TAMPINES ST 34', 'TAMPINES AVE 8',
                'TAMPINES ST 82', 'TAMPINES ST 33', 'TAMPINES ST 84',
                'TAMPINES ST 45', 'TAMPINES ST 71', 'SIMEI ST 4', 'TAMPINES ST 91',
                'TAMPINES ST 24', 'TAMPINES ST 72', 'TAMPINES ST 32',
                'TAMPINES CTRL 7', 'SIMEI RD', 'LOR 2 TOA PAYOH', 'KIM KEAT AVE',
                'UPP ALJUNIED LANE', 'LOR 1 TOA PAYOH', 'LOR 6 TOA PAYOH',
                'LOR 5 TOA PAYOH', 'LOR 7 TOA PAYOH', 'TOA PAYOH EAST',
                'LOR 3 TOA PAYOH', 'LOR 4 TOA PAYOH', 'TOA PAYOH CTRL',
                'POTONG PASIR AVE 2', 'LOR 8 TOA PAYOH', 'POTONG PASIR AVE 1',
                'JOO SENG RD', 'KIM KEAT LINK', 'MARSILING RISE', 'MARSILING DR',
                'WOODLANDS ST 31', 'WOODLANDS ST 41', 'WOODLANDS DR 16',
                'WOODLANDS ST 83', 'WOODLANDS ST 82', 'WOODLANDS CIRCLE',
                'WOODLANDS DR 60', 'WOODLANDS ST 13', 'WOODLANDS CRES',
                'WOODLANDS ST 81', 'WOODLANDS AVE 6', 'WOODLANDS DR 40',
                'WOODLANDS DR 52', 'WOODLANDS DR 70', 'WOODLANDS DR 44',
                'WOODLANDS DR 50', 'WOODLANDS DR 42', 'WOODLANDS DR 62',
                'WOODLANDS AVE 1', 'WOODLANDS DR 14', 'WOODLANDS DR 53',
                'WOODLANDS RING RD', 'WOODLANDS ST 11', 'WOODLANDS DR 75',
                'WOODLANDS AVE 5', 'WOODLANDS ST 32', 'YISHUN RING RD',
                'YISHUN AVE 5', 'YISHUN AVE 6', 'YISHUN ST 22', 'YISHUN CTRL',
                'YISHUN AVE 2', 'YISHUN AVE 4', 'YISHUN ST 21', 'YISHUN ST 11',
                'YISHUN AVE 11', 'YISHUN AVE 3', 'YISHUN AVE 9', 'YISHUN ST 61',
                'YISHUN ST 72', 'YISHUN ST 81', 'ANG MO KIO ST 32',
                'BEDOK NTH ST 4', 'BT BATOK WEST AVE 7', 'JLN RUMAH TINGGI',
                'TELOK BLANGAH WAY', 'TIONG BAHRU RD', 'TELOK BLANGAH RISE',
                'HENDERSON CRES', 'BT PURMEI RD', 'SPOTTISWOODE PK RD',
                'LOMPANG RD', 'SELEGIE RD', 'KELANTAN RD', 'KRETA AYER RD',
                'CHOA CHU KANG ST 53', 'CLEMENTI AVE 6', 'CLEMENTI ST 13',
                'SIMS PL', 'EUNOS RD 5', 'SIMS AVE', 'BUANGKOK CRES',
                'HOUGANG ST 31', 'JURONG EAST AVE 1', 'JURONG WEST ST 24',
                'YUNG PING RD', 'LOR LIMAU', 'TOWNER RD', 'NTH BRIDGE RD',
                'KG ARANG RD', 'DORSET RD', "ST. GEORGE'S LANE", 'PASIR RIS ST 13',
                'PUNGGOL FIELD WALK', 'PUNGGOL EAST', "QUEEN'S CL", "C'WEALTH AVE",
                'CLARENCE LANE', 'DOVER CL EAST', 'ANCHORVALE RD',
                'ANCHORVALE LANE', 'FERNVALE LANE', 'LOR LEW LIAN',
                'SERANGOON NTH AVE 2', 'TAMPINES AVE 5', 'TAMPINES CTRL 1',
                'SIMEI ST 5', 'TAMPINES AVE 7', 'MARSILING LANE',
                'WOODLANDS AVE 4', 'WOODLANDS DR 73', 'WOODLANDS DR 72',
                'MARSILING RD', 'YISHUN ST 71', 'YISHUN ST 20', 'ANG MO KIO AVE 9',
                'ANG MO KIO AVE 2', 'CHAI CHEE DR', 'SIN MING RD', 'MOH GUAN TER',
                'BT MERAH CTRL', "QUEEN'S RD", 'EMPRESS RD', 'JLN KUKOH',
                'VEERASAMY RD', 'WATERLOO ST', 'KLANG LANE', 'CHOA CHU KANG ST 52',
                'CHOA CHU KANG LOOP', 'CHOA CHU KANG ST 54', 'CHOA CHU KANG NTH 7',
                'CLEMENTI ST 12', "C'WEALTH AVE WEST", 'GEYLANG EAST CTRL',
                'GEYLANG SERAI', 'PIPIT RD', 'YUAN CHING RD', 'JURONG WEST ST 72',
                'JURONG WEST ST 62', 'KG KAYU RD', 'WHAMPOA WEST', "JLN MA'MOR",
                'CAMBRIDGE RD', 'PUNGGOL RD', 'SEMBAWANG CRES', 'SEMBAWANG WAY',
                'TAMPINES AVE 9', 'SIMEI ST 2', 'TOA PAYOH NTH', 'JLN DAMAI',
                'BT BATOK ST 22', 'DELTA AVE', 'QUEEN ST', 'DAKOTA CRES',
                'BUANGKOK LINK', 'UPP SERANGOON RD', "KING GEORGE'S AVE",
                'LOR 3 GEYLANG', 'JELLICOE RD', 'PASIR RIS ST 41',
                'WOODLANDS AVE 3', 'WOODLANDS DR 71', 'TAMAN HO SWEE',
                'EVERTON PK', 'ROWELL RD', 'SMITH ST', 'CLEMENTI ST 14',
                'YUNG HO RD', 'KENT RD', 'POTONG PASIR AVE 3', 'YISHUN AVE 7',
                'BISHAN ST 11', 'INDUS RD', 'SAGO LANE', 'NEW MKT RD',
                'CHANDER RD', 'OLD AIRPORT RD', 'WOODLANDS AVE 9', 'KIM PONG RD',
                'BUFFALO RD', 'CANBERRA LINK', 'BAIN ST', 'JLN DUA', 'OWEN RD',
                'TESSENSOHN RD', 'GHIM MOH LINK', 'MARSILING CRES',
                'ANG MO KIO ST 11', 'SILAT AVE', 'KIM CHENG ST', 'MOULMEIN RD',
                'CLEMENTI ST 11', 'YISHUN CTRL 1', 'JLN BERSEH', 'FRENCH RD',
                'BT MERAH LANE 1', 'SIMEI LANE', 'JOO CHIAT RD', 'TAO CHING RD',
                'CLEMENTI AVE 1', 'YISHUN ST 41', 'TELOK BLANGAH ST 31', 'ZION RD',
                'JLN KAYU', 'LOR 1A TOA PAYOH', 'PUNGGOL WALK',
                'SENGKANG WEST WAY', 'BUANGKOK GREEN', 'PUNGGOL WAY',
                'YISHUN ST 31', 'TECK WHYE CRES', 'MONTREAL LINK',
                'UPP SERANGOON CRES', 'SUMANG LINK', 'SENGKANG EAST AVE',
                'YISHUN AVE 1', 'ANCHORVALE CRES', 'YUNG KUANG RD',
                'ANCHORVALE ST', 'TAMPINES CTRL 8', 'YISHUN ST 51',
                'UPP SERANGOON VIEW', 'TAMPINES AVE 1', 'BEDOK RESERVOIR CRES',
                'ANG MO KIO ST 61', 'DAWSON RD', 'FERNVALE ST', 'HOUGANG ST 32',
                'TAMPINES ST 86', 'HENDERSON RD', 'SUMANG WALK',
                'CHOA CHU KANG AVE 7', 'KEAT HONG CL', 'JURONG WEST CTRL 3',
                'KEAT HONG LINK', 'ALJUNIED AVE 2', 'CANBERRA CRES', 'SUMANG LANE',
                'CANBERRA ST', 'ANG MO KIO ST 44', 'ANG MO KIO ST 51',
                'BT BATOK EAST AVE 6', 'BT BATOK WEST AVE 9', 'GEYLANG EAST AVE 2',
                'MARINE PARADE CTRL', 'CANBERRA WALK', 'WOODLANDS RISE',
                'TAMPINES ST 61', 'YISHUN ST 43']
            # -----New Data inputs from the user for predicting the resale price-----
            street_name = st.selectbox("Street Name", options=streets)
            block = st.text_input("Block Number")
            floor_area_sqm = st.number_input('Floor Area (Per Square Meter)', min_value=1.0, max_value=500.0)
            lease_commence_date = st.number_input('Lease Commence Date')
            storey_range = st.text_input("Storey Range (Format: 'Value1' TO 'Value2')")

            # -----Submit Button for PREDICT RESALE PRICE-----
            submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")

            if submit_button is not None:
                with open(r"model.pkl", 'rb') as file:
                    loaded_model = pickle.load(file)
                with open(r'scaler.pkl', 'rb') as f:
                    scaler_loaded = pickle.load(f)

                # -----Calculating lease_remain_years using lease_commence_date-----
                lease_remain_years = 99 - (2023 - lease_commence_date)

                # -----Calculating median of storey_range to make our calculations quite comfortable-----
                split_list = re.split(' TO | To | to', storey_range)
                float_list = [float(i) for i in split_list]
                storey_median = statistics.median(float_list)

                # -----Getting the address by joining the block number and the street name-----
                origin = []

                # -----Getting the address by joining the block number and the street name-----
                address = block + " " + street_name
                data = pd.read_csv('df_coordinates.csv')
                
 

            # Filter the DataFrame based on the block number and road name
                filtered_data = data[(data['blk_no'] == block) & (data['road_name'] == street_name)]


            # Get latitude and longitude from the filtered data
                # latitude = filtered_data.iloc[0]['latitude']
                # longitude = filtered_data.iloc[0]['longitude']
                # origin.append((latitude, longitude))

                # -----Appending the Latitudes and Longitudes of the MRT Stations-----
                # Latitudes and Longitudes are been appended in the form of a tuple  to that list
                mrt_lat = mrt_location['latitude']
                mrt_long = mrt_location['longitude']
                list_of_mrt_coordinates = []
                for lat, long in zip(mrt_lat, mrt_long):
                    list_of_mrt_coordinates.append((lat, long))

                # -----Getting distance to nearest MRT Stations (Mass Rapid Transit System)-----
                list_of_dist_mrt = []
                for destination in range(0, len(list_of_mrt_coordinates)):
                    list_of_dist_mrt.append(geodesic(origin, list_of_mrt_coordinates[destination]).meters)
                shortest = (min(list_of_dist_mrt))
                min_dist_mrt = shortest
                list_of_dist_mrt.clear()

                # -----Getting distance from CDB (Central Business District)-----
                cbd_dist = geodesic(origin, (1.2830, 103.8513)).meters  # CBD coordinates

                # -----Sending the user enter values for prediction to our model-----
                new_sample = np.array(
                    [[cbd_dist, min_dist_mrt, np.log(floor_area_sqm), lease_remain_years, np.log(storey_median)]])
                new_sample = scaler_loaded.transform(new_sample[:, :5])
                new_pred = loaded_model.predict(new_sample)[0]
                st.write('## :green[Predicted resale price:] ', round(np.exp(new_pred), 2))

    except Exception as e:
        st.write("Enter the above values to get the predicted resale price of the flat")
