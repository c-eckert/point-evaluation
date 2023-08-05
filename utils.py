import streamlit as st
import cv2
import cv2.aruco as aruco
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
import requests
import base64
import pandas as pd

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


# FUNCTIONS
@st.cache_data
def convert_image_to_cv2(bytes_data):
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return cv2_img, rgb_img


@st.cache_data
def detect_draw_aruco_marker(img):
    id = 0
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        aruco.drawDetectedMarkers(img, corners, ids)
        if len(ids) == 1:
            id = ids[0][0]
    return id, img

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
def draw_row(img, df_row, row_name):
    min_x = int(df_row["x"].min() - df_row["w"].min()/2)
    min_y = int(df_row["y"].min() - df_row["h"].min()/2)
    max_x = int(df_row["x"].max() + df_row["w"].min()/2)
    max_y = int(df_row["y"].max() + df_row["h"].min()/2)

    cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
    cv2.putText(img, row_name, (max_x + 5, max_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return img

@st.cache_data
def do_kmeans(df, inertia_th, cluster_list=None):
    y_coordinates = df['y'].values.reshape(-1, 1)
    if cluster_list:
        kmeans = KMeans(n_clusters=len(cluster_list))
        kmeans.fit(y_coordinates)
    else:
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(y_coordinates)
            if kmeans.inertia_ < inertia_th:
                n_clusters = i
                break
        cluster_list = list(map(str, range(1, n_clusters + 1)))
    
    # Cluster nach der Höhe sortieren
    cluster_centers = kmeans.cluster_centers_
    order = cluster_centers.argsort(axis=0).flatten()
    rank = order.argsort()
    df['cluster'] = [cluster_list[i] for i in rank[kmeans.labels_]]

    return cluster_list, df

@st.cache_data
def draw_points(img, df):
    for idx in df.index:
        x0 = int(df["x"][idx] - (df["w"][idx]/2))
        y0 = int(df["y"][idx] - (df["h"][idx]/2))
        x1 = int(df["x"][idx] + (df["w"][idx]/2))
        y1 = int(df["y"][idx] + (df["h"][idx]/2))
        cv2.rectangle(img, (x0, y0), (x1, y1), CLASS_TO_COLOR[df["class"][idx]], 2)
        cv2.putText(img, f"{int(df['conf'][idx]*100)}%", (x0 + 1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img



@st.cache_data
def generate_csv_from_list(dictionary):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    df = pd.DataFrame(dictionary)
    return df.to_csv().encode('utf-8')

@st.cache_data
def generate_csv_from_df(df):
    return df.to_csv().encode('utf-8')


def initialize_global_csv():
    df = pd.DataFrame(columns=['Tagespunkte', 'Rallypunkte'], index=range(200))
    df.to_csv('data.csv')

def deine_liste(id, punkte, liste):
    st.header("Deine Liste")
    input_col1, input_col2 = st.columns(2)
    with input_col1:
        id_number = st.number_input('ID der Person', min_value=0, value=id, step=1)
    with input_col2:
        pkte_number = st.number_input('Punktzahl', min_value=0, value=punkte, step=1)


    overlist_col1, overlist_col2 = st.columns(2)
    with overlist_col1:
        if st.button('➕'):
            st.session_state[liste].append({"id": id_number, "punkte": pkte_number})

    with overlist_col2:
        if st.button('➖'):
            if len(st.session_state[liste]) > 0:
                st.session_state[liste].pop()

    st.table(st.session_state[liste])

    underlist_col1, underlist_col2 = st.columns(2)
    with underlist_col1:
        if st.button('🗑️ Leeren', key="reset my list"):
            st.session_state[liste] = []
            st.experimental_rerun()

    with underlist_col2:
        st.download_button(
            label="⬇️ Exportieren",
            key="export my list to csv",
            data=generate_csv_from_list(st.session_state[liste]),
            file_name=f'punkte_{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv',
            mime='text/csv',
        )

def gemeinsame_liste(meine_liste, meine_spalte):
    st.header("Gemeinsame Liste")

    st.session_state.password = st.text_input("Password eingeben", type="password")

    if st.session_state.password == PASSWORD:
        st.success('Password korrekt')
        st.session_state.button_disabled = False
    else:
        st.info('Password eingeben, um gemeinsame Liste zu bearbeiten')

    if st.button('🔀 Meine und gemeinsame Liste zusammenführen', disabled=st.session_state.button_disabled):
        df = pd.read_csv('data.csv', index_col=0)
        for i in st.session_state[meine_liste]:
            df[meine_spalte][i['id']]=i['punkte']
        df.to_csv('data.csv')

    st.write(pd.read_csv('data.csv', index_col=0))

    sharedlist_col1, sharedlist_col2 = st.columns(2)
    with sharedlist_col1:
        if st.button('🗑️ Liste löschen', key="reset shared list", disabled=st.session_state.button_disabled):
            initialize_global_csv()
            st.experimental_rerun()

    with sharedlist_col2:
        st.download_button(
            label="⬇️ Exportieren",
            key="export shared list to csv",
            data=generate_csv_from_df(pd.read_csv('data.csv', index_col=0)),
            file_name=f'gemeinsame_punktliste_{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv',
            mime='text/csv',
        )
