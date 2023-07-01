import streamlit as st
import cv2
import numpy as np
#from roboflow import Roboflow
import base64

#rf = Roboflow(api_key=st.secrets["RF_API_KEY"])
#project = rf.workspace().project("sticky-dot-counter")
#model = project.version(1).model


img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    retval, buffer = cv2.imencode('.jpg', cv2_img)
    img_str = base64.b64encode(buffer)

 #   st.write(model.predict(img_str, confidence=40, overlap=30).json())

    # Check the type of cv2_img:
    # Should output: <class 'numpy.ndarray'>
    st.write(type(cv2_img))

    # Check the shape of cv2_img:
    # Should output shape: (height, width, channels)
    st.write(cv2_img.shape)



st.markdown(
    "This demo uses a model and code from "
    "https://github.com/robmarkcole/object-detection-app. "
    "Many thanks to the project."
)