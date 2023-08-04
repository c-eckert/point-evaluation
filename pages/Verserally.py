import streamlit as st
import pandas as pd
import os
import utils

st.set_page_config(
    page_title="Verserally",
    page_icon="👋",
)

with open("streamlit.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# SESSION STATES

if 'rallyliste' not in st.session_state:
    st.session_state.rallyliste = []

if 'password' not in st.session_state:
    st.session_state.password = ""

if 'button_disabled' not in st.session_state:
    st.session_state.button_disabled = True


file_path = 'data.csv'  # Replace with the actual file path

if not os.path.isfile(file_path):
    utils.initialize_global_csv()

id = 0
pt_verserally = 0

# WEBSITE
st.header("Verserally")

inertia_th = st.sidebar.number_input("kmeans.inertia", min_value=0, max_value=5000, value=800, step=1)

img_file_buffer = st.camera_input("Hier die Bewertungskarte fotografieren", key="camera")

if img_file_buffer:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img, rgb_img = utils.convert_image_to_cv2(bytes_data)
    
    # ARUCO
    id, img = utils.detect_draw_aruco_marker(rgb_img)

    # Point Detection
    d = utils.detect_points(cv2_img)
    if d["points"]:
        df = pd.DataFrame(data=d)
        img = utils.draw_points(img, df)

        cluster_list, df = utils.do_kmeans(df, inertia_th)

        for row_name in cluster_list:
            df_row = df[df['cluster'] == row_name]
            img = utils.draw_row(img, df_row, row_name)
            
            if int(df_row.shape[0]) == 5:
                pt_verserally += 10
            else:
                pt_verserally += int(df_row.shape[0])

        st.image(img)
        st.write(f"{pt_verserally} Punkte wurden erkannt...")


utils.deine_liste(id, pt_verserally, "rallyliste")

st.write("---") 

utils.gemeinsame_liste("rallyliste", "Rallypunkte")