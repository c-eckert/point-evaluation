import streamlit as st
import cv2
import numpy as np
import requests
import base64

#rf = Roboflow(api_key=st.secrets["RF_API_KEY"])
#project = rf.workspace().project("sticky-dot-counter")
#model = project.version(1).model

ROBOFLOW_MODEL = "sticky-dot-counter"
ROBOFLOW_API_KEY = st.secrets["RF_API_KEY"]

upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL,
    "?access_token=",
    ROBOFLOW_API_KEY,
    "&format=image",
    "&stroke=5"
])


img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    retval, buffer = cv2.imencode('.jpg', cv2_img)
    img_str = base64.b64encode(buffer)

    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True).raw

    st.write(type(cv2_img))

    # Parse result image
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imshow('image', image)
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