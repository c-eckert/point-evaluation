import streamlit as st
import cv2
import cv2.aruco as aruco
import numpy as np
import requests
import base64
import pandas as pd
from datetime import datetime
import os

ROBOFLOW_MODEL = st.secrets["RF_MODEL"]
ROBOFLOW_API_KEY = st.secrets["RF_API_KEY"]
PASSWORD = st.secrets["PASSWORD"]

CLASS_TO_COLOR = {
    "red-dot": (255, 0, 0),
    "yellow-dot": (255, 255, 0),
    "green-dot": (0, 255, 0)
}

CLASS_TO_VALUE = {
    "red-dot": 0,
    "yellow-dot": 1,
    "green-dot": 2
}
ROW_NAMES = [
    "Vers",
    "Mitarbeit",
    "Benehmen",
    "Zimmer"
]

st.write('''<style>
[data-testid="column"] {
    width: calc(33.3333% - 1rem) !important;
    flex: 1 1 calc(33.3333% - 1rem) !important;
    min-width: calc(33% - 1rem) !important;
}
</style>''', unsafe_allow_html=True)

# FUNCTIONS
@st.cache_data
def detect_aruco_marker(img):
    id = 0
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is not None:
        aruco.drawDetectedMarkers(rgb_img, corners, ids)
        if len(ids) == 1:
            id = ids[0][0]
    return id

@st.cache_data
def detect_points(img):
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)
    upload_url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}/1?api_key={ROBOFLOW_API_KEY}"

    resp = requests.post(
        upload_url, 
        data=img_str, 
        headers={"Content-Type": "application/x-www-form-urlencoded"})

    if resp.status_code == 200:
        detection = resp.json()
        
        d = {
            "x": [i["x"] for i in detection["predictions"]], 
            "y": [i["y"] for i in detection["predictions"]], 
            "w": [i["width"] for i in detection["predictions"]], 
            "h": [i["height"] for i in detection["predictions"]], 
            "conf": [i["confidence"] for i in detection["predictions"]], 
            "class": [i["class"] for i in detection["predictions"]],
            "points": [CLASS_TO_VALUE[i["class"]] for i in detection["predictions"]]
        }
        return d

@st.cache_data
def draw_rows(df, img):
    pt_sums = dict.fromkeys(ROW_NAMES, 0)
    for i, row_name in enumerate(ROW_NAMES):
        df_row = df.iloc[i*5:(i+1)*5]
        if not df_row.empty:
            points = int(df_row["points"].sum())
            pt_sums[row_name] = points
            
            min_x = int(df_row["x"].min() - df_row["w"].min()/2)
            min_y = int(df_row["y"].min() - df_row["h"].min()/2)
            max_x = int(df_row["x"].max() + df_row["w"].min()/2)
            max_y = int(df_row["y"].max() + df_row["h"].min()/2)

            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
            cv2.putText(img, row_name, (max_x + 5, max_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return pt_sums

@st.cache_data
def draw_points(img):
    for idx in df.index:
        x0 = int(df["x"][idx] - (df["w"][idx]/2))
        y0 = int(df["y"][idx] - (df["h"][idx]/2))
        x1 = int(df["x"][idx] + (df["w"][idx]/2))
        y1 = int(df["y"][idx] + (df["h"][idx]/2))
        cv2.rectangle(img, (x0, y0), (x1, y1), CLASS_TO_COLOR[df["class"][idx]], 2)
        cv2.putText(img, f"{int(df['conf'][idx]*100)}%", (x0 + 1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

@st.cache_data
def generate_csv_from_list(dictionary):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    df = pd.DataFrame(dictionary)
    return df.to_csv().encode('utf-8')

@st.cache_data
def generate_csv_from_df(df):
    return df.to_csv().encode('utf-8')

@st.cache_data
def initialize_global_csv():
    df = pd.DataFrame(columns=['Punkte'], index=range(200))
    df.to_csv('data.csv')


# SESSION STATES

if 'liste' not in st.session_state:
    st.session_state.liste = []

if 'password' not in st.session_state:
    st.session_state.password = ""

file_path = 'data.csv'  # Replace with the actual file path

if not os.path.isfile(file_path):
    initialize_global_csv()

# WEBSITE
st.header("Auswertung")

img_file_buffer = st.camera_input("Hier die Bewertungskarte fotografieren", key="camera")

FRAME_WINDOW = st.image([])

id = 0
pt_ges = 0

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    
    # ARUCO
    id = detect_aruco_marker(rgb_img)

    # Point Detection
    d = detect_points(cv2_img)
    if d:
        df = pd.DataFrame(data=d)
        df = df.sort_values(by=["y"])

        pt_sums = draw_rows(df, rgb_img) # Reihenerkennung
        draw_points(rgb_img) # Punkteerkennung
        FRAME_WINDOW.image(rgb_img)
        
        pt_ges = pt_sums["Vers"] * 3 + pt_sums["Mitarbeit"] * 3 + pt_sums["Benehmen"] * 3 + pt_sums["Zimmer"]
        
        st.write(f"{df.shape[0]} von 20 Punkten wurden erkannt...")

        with st.expander(f"Gesamtpunktzahl: {pt_ges} (siehe Erläuterung)"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                Vers:  
                Mitarbeit:  
                Benehmen:  
                Zimmer:
                #### Gesamtpunkte:
                """)
            
            with col2:
                st.markdown(f"""
                &emsp; **{pt_sums["Vers"]}** * 3  
                &plus;&ensp; **{pt_sums["Mitarbeit"]}** * 3  
                &plus;&ensp;  **{pt_sums["Benehmen"]}** * 3  
                &plus;&ensp;  **{pt_sums["Zimmer"]}** * 1
                #### =&ensp; {pt_ges}
                """)


st.write("---") 
input_col1, input_col2 = st.columns(2)
with input_col1:
    id_number = st.number_input('ID der Person', min_value=0, value=id, step=1)
with input_col2:
    pkte_number = st.number_input('Punktzahl', min_value=0, value=pt_ges, step=1)

st.header("Deine Liste")

mylist_col1, mylist_col2 = st.columns(2)
with mylist_col1:
    if st.button('➕'):
        st.session_state.liste.append({"id": id_number, "punkte": pkte_number})
        st.success('Punkte hinzugefügt')
    
    if st.button('➖'):
        if len(st.session_state.liste) > 0:
            st.session_state.liste.pop()

    if st.button('🗑️ alles löschen', key="reset my list"):
        st.session_state.liste = []

    st.download_button(
        label="⬇️ Exportieren",
        key="export my list to csv",
        data=generate_csv_from_list(st.session_state.liste),
        file_name=f'punkte_{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv',
        mime='text/csv',
    )

with mylist_col2:
    st.table(st.session_state.liste)


st.write("---") 

st.header("Gemeinsame Liste")

st.session_state.password = st.text_input("Password eingeben", type="password")
button_disabled = True
if st.session_state.password == PASSWORD:
    st.success('Password korrekt')
    button_disabled = False
else:
    st.info('Password eingeben, um gemeinsame Liste zu bearbeiten')

sharedlist_col1, sharedlist_col2 = st.columns(2)

with sharedlist_col1:
    if st.button('🔀 Zusammenführen', disabled=button_disabled):
        df = pd.read_csv('data.csv', index_col=0)
        for i in st.session_state.liste:
            df['Punkte'][i['id']]=i['punkte']
        df.to_csv('data.csv')

    if st.button('🗑️ Liste löschen', key="reset shared list", disabled=button_disabled):
        initialize_global_csv()

    st.download_button(
        label="⬇️ Exportieren",
        key="export shared list to csv",
        data=generate_csv_from_df(pd.read_csv('data.csv', index_col=0)),
        file_name=f'gemeinsame_punktliste_{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv',
        mime='text/csv',
    )

with sharedlist_col2:
    st.write(pd.read_csv('data.csv', index_col=0))
