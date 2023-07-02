import streamlit as st
import cv2
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

img_file_buffer = st.camera_input("Take a picture", key="camera")

st.write("Detektion...")
FRAME_WINDOW = st.image([])





if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    retval, buffer = cv2.imencode('.jpg', cv2_img)
    img_str = base64.b64encode(buffer)

    resp = requests.post(
        upload_url, 
        data=img_str, 
        headers={"Content-Type": "application/x-www-form-urlencoded"})

    if resp.status_code == 200:
        detection = resp.json()
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        data_x = [i["x"] for i in detection["predictions"]]
        data_y = [i["y"] for i in detection["predictions"]]
        data_w = [i["width"] for i in detection["predictions"]]
        data_h = [i["height"] for i in detection["predictions"]]
        data_conf = [i["confidence"] for i in detection["predictions"]]
        data_class = [i["class"] for i in detection["predictions"]]
        data_points = [CLASS_TO_VALUE[i["class"]] for i in detection["predictions"]]
        d = {
            "x": data_x, 
            "y": data_y, 
            "w": data_w, 
            "h": data_h, 
            "conf": data_conf, 
            "class": data_class,
            "points": data_points
        }
        df = pd.DataFrame(data=d)
        df = df.sort_values(by=["y"])
        pt_vers = df["points"].iloc[:5].sum()
        pt_mitarbeit = df["points"].iloc[5:10].sum()
        pt_benehmen = df["points"].iloc[10:15].sum()
        pt_zimmer = df["points"].iloc[15:20].sum()
        pt_ges = pt_vers * 3 + pt_mitarbeit * 3 + pt_benehmen * 1
        
        st.write(f"{df.shape[0]} von 20 Punkten erkannt...")
        col1, col2 = st.columns([1, 2])
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
            &emsp; **{pt_vers}** * 3  
            &plus;&ensp; **{pt_mitarbeit}** * 3  
            &plus;&ensp;  **{pt_benehmen}** * 3  
            &plus;&ensp;  **{pt_zimmer}** * 1
            #### =&ensp; {pt_ges}
            """)

        if st.button('Save detection'):
            st.session_state.liste.append({"id": 12, "punkte": int(pt_ges)})


        st.write(detection)
        for idx in df.index:
            print(idx)
            #for pred in detection["predictions"]:
            x0 = int(df["x"][idx] - (df["w"][idx]/2))
            y0 = int(df["y"][idx] - (df["h"][idx]/2))
            x1 = int(df["x"][idx] + (df["w"][idx]/2))
            y1 = int(df["y"][idx] + (df["h"][idx]/2))

            rgb_img = cv2.rectangle(rgb_img, (x0, y0), (x1, y1), CLASS_TO_COLOR[df["class"][idx]], 2)

        FRAME_WINDOW.image(rgb_img)

st.write(st.session_state.liste)