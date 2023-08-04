import streamlit as st
import pandas as pd
import os
import utils

st.set_page_config(
    page_title="Gesamtbewertung",
    page_icon="👋",
)

with open("streamlit.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# SESSION STATES

if 'liste' not in st.session_state:
    st.session_state.liste = []

if 'password' not in st.session_state:
    st.session_state.password = ""

if 'button_disabled' not in st.session_state:
    st.session_state.button_disabled = True


file_path = 'data.csv'  # Replace with the actual file path

if not os.path.isfile(file_path):
    utils.initialize_global_csv()

id = 0
pt_gesamtbewertung = 0

# WEBSITE
st.header("Gesamtbewertung")

faktor_vers = st.sidebar.slider("Gewichtung Vers", min_value=0, max_value=5, value=3, step=1)
faktor_mitarbeit = st.sidebar.slider("Gewichtung Mitarbeit", min_value=0, max_value=5, value=3, step=1)
faktor_benehmen = st.sidebar.slider("Gewichtung Benehmen", min_value=0, max_value=5, value=3, step=1)
faktor_zimmer = st.sidebar.slider("Gewichtung Zimmer", min_value=0, max_value=5, value=1, step=1)


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

        cluster_list, df = utils.do_kmeans(df, 800, utils.ROW_NAMES)
        pt_sums = dict.fromkeys(cluster_list, 0)
        for row_name in cluster_list:
            df_row = df[df['cluster'] == row_name]
            points = int(df_row["points"].sum())
            pt_sums[row_name] = points
            img = utils.draw_row(img, df_row, row_name)
        
        st.image(img)
        
        pt_gesamtbewertung = pt_sums["Vers"]*faktor_vers + pt_sums["Mitarbeit"]*faktor_mitarbeit + pt_sums["Benehmen"]*faktor_benehmen + pt_sums["Zimmer"]*faktor_zimmer
        
        st.write(f"{df.shape[0]} von 20 Punkten wurden erkannt...")

        with st.expander(f"Gesamtpunktzahl: {pt_gesamtbewertung} (siehe Erläuterung)"):
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
                &emsp; **{pt_sums["Vers"]}** * {faktor_vers}  
                &plus;&ensp; **{pt_sums["Mitarbeit"]}** * {faktor_mitarbeit}  
                &plus;&ensp;  **{pt_sums["Benehmen"]}** * {faktor_benehmen}  
                &plus;&ensp;  **{pt_sums["Zimmer"]}** * {faktor_zimmer}  
                #### =&ensp; {pt_gesamtbewertung}
                """)



utils.deine_liste(id, pt_gesamtbewertung, "liste")

st.write("---") 

utils.gemeinsame_liste("liste", "Tagespunkte")

