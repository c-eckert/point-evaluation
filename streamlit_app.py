import streamlit as st
#import cv2
#import numpy as np
import av
#import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode



score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)



def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    h, w = image.shape[:2]
    return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    #rtc_configuration={
    #    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
    #    "iceTransportPolicy": "relay",
    #},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.checkbox("Show the detected labels", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()


st.markdown(
    "This demo uses a model and code from "
    "https://github.com/robmarkcole/object-detection-app. "
    "Many thanks to the project."
)