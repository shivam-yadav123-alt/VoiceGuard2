import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("WebRTC Test")

webrtc_streamer(
    key="test",
    media_stream_constraints={
        "audio": True,
        "video": False,
    },
)
