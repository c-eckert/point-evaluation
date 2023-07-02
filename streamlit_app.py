import streamlit as st
import cv2
import cv2.aruco as aruco
import numpy as np
import requests
import base64
import pandas as pd

ROBOFLOW_MODEL = st.secrets["RF_MODEL"]
ROBOFLOW_API_KEY = st.secrets["RF_API_KEY"]

if 'liste' not in st.session_state:
    st.session_state.liste = []

upload_url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}/1?api_key={ROBOFLOW_API_KEY}"

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

img_file_buffer = st.camera_input("Take a picture", key="camera")

FRAME_WINDOW = st.image([])

id = 0
pt_ges = 0

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    
    # ARUCO
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is not None:
        aruco.drawDetectedMarkers(rgb_img, corners, ids)
        if len(ids) == 1:
            id = ids[0][0]

    # Point Detection
    retval, buffer = cv2.imencode('.jpg', cv2_img)
    img_str = base64.b64encode(buffer)

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
        df = pd.DataFrame(data=d)
        df = df.sort_values(by=["y"])

        # Reihenerkennung
        pt_sums = dict.fromkeys(ROW_NAMES, 0)
        for i, row_name in enumerate(ROW_NAMES):
            df_row = df.iloc[i*5:(i+1)*5]
            if not df_row.empty:
                points = int(df_row["points"].sum())
                min_x = int(df_row["x"].min() - df_row["w"].min()/2)
                min_y = int(df_row["y"].min() - df_row["h"].min()/2)
                max_x = int(df_row["x"].max() + df_row["w"].min()/2)
                max_y = int(df_row["y"].max() + df_row["h"].min()/2)

                pt_sums[row_name] = points

                rgb_img = cv2.rectangle(rgb_img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
                rgb_img = cv2.putText(rgb_img, row_name, (max_x + 5, max_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Punkteerkennung
        for idx in df.index:
            x0 = int(df["x"][idx] - (df["w"][idx]/2))
            y0 = int(df["y"][idx] - (df["h"][idx]/2))
            x1 = int(df["x"][idx] + (df["w"][idx]/2))
            y1 = int(df["y"][idx] + (df["h"][idx]/2))
            rgb_img = cv2.rectangle(rgb_img, (x0, y0), (x1, y1), CLASS_TO_COLOR[df["class"][idx]], 2)
            rgb_img = cv2.putText(rgb_img, f"{int(df['conf'][idx]*100)}%", (x0 + 1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
        FRAME_WINDOW.image(rgb_img)
        
        pt_ges = pt_sums["Vers"] * 3 + pt_sums["Mitarbeit"] * 3 + pt_sums["Benehmen"] * 3 + pt_sums["Zimmer"]
        
        st.write(f"{df.shape[0]} von 20 Punkten wurden erkannt...")
        #col1, col2 = st.columns(2)
        #with col1:
        #    st.markdown(f"""
        #    Vers:  
        #    Mitarbeit:  
        #    Benehmen:  
        #    Zimmer:
        #    #### Gesamtpunkte:
        #    """)
        
        #with col2:
        #    st.markdown(f"""
        #    &emsp; **{pt_sums["Vers"]}** * 3  
        #    &plus;&ensp; **{pt_sums["Mitarbeit"]}** * 3  
        #    &plus;&ensp;  **{pt_sums["Benehmen"]}** * 3  
        #    &plus;&ensp;  **{pt_sums["Zimmer"]}** * 1
        #    #### =&ensp; {pt_ges}
        #    """)
        with st.expander(f"Gesamtpunktzahl: {pt_ges} (siehe Erläuterung)"):
            st.markdown(f"""
                | Kategorie    | Punkte     |
                |--------------|------------|
                | Vers         | &emsp; **{pt_sums["Vers"]}** * 3 |
                | Mitarbeit    | &plus;&ensp; **{pt_sums["Mitarbeit"]}** * 3 |
                | Benehmen     | &plus;&ensp;  **{pt_sums["Benehmen"]}** * 3 |
                | Zimmer       | &plus;&ensp;  **{pt_sums["Zimmer"]}** * 1 |
                | **Gesamtpunkte** | =&ensp; **{pt_ges}** |
                """)





st.write("---") 
input_col1, input_col2 = st.columns(2)
with input_col1:
    id_number = st.number_input('ID der Person', min_value=0, value=id, step=1)
with input_col2:
    pkte_number = st.number_input('Punktzahl', min_value=0, value=pt_ges, step=1)


but_col1, but_col2, but_col3, but_col4 = st.columns(4)
with but_col1:
    if st.button('Add detection to list'):
        st.session_state.liste.append({"id": id_number, "punkte": pkte_number})

with but_col2:
    if st.button('Delete last entry'):
        st.session_state.liste.pop()

with but_col3:
    if st.button('Clear all detections'):
        st.session_state.liste = []

with but_col4:
    if st.button('Commit list'):
        st.write("Not working")

st.table(st.session_state.liste)