import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import os

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

MODEL_PATH = "best_helmet_model.pt"

model = YOLO(MODEL_PATH)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp{
    background: linear-gradient(to right, #071226, #111827);
    color:white;
}

.main-title{
    text-align:center;
    color:#38bdf8;
    font-size:55px;
    font-weight:800;
    margin-top:20px;
}

.main-subtitle{
    text-align:center;
    color:lightgray;
    font-size:22px;
    margin-bottom:40px;
}

.upload-box{
    padding:25px;
    border-radius:20px;
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.1);
    margin-bottom:25px;
}

.result-box{
    padding:25px;
    border-radius:20px;
    background:#111827;
    border:1px solid #38bdf8;
    margin-top:20px;
}

.success-box{
    background:rgba(34,197,94,0.2);
    border:1px solid #22c55e;
    padding:15px;
    border-radius:12px;
    margin-top:20px;
    color:#bbf7d0;
}

.metric-bar{
    width:100%;
    height:18px;
    background:#333;
    border-radius:10px;
    overflow:hidden;
    margin-top:10px;
}

.metric-fill{
    height:100%;
    background:linear-gradient(to right,#22c55e,#38bdf8);
}

@media (max-width:768px){
    .main-title{
        font-size:38px !important;
    }

    .main-subtitle{
        font-size:16px;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown('<h1 class="main-title">🪖 Helmet Detection System</h1>', unsafe_allow_html=True)

st.markdown(
    '<div class="main-subtitle">YOLOv8 Powered Smart Safety Detection</div>',
    unsafe_allow_html=True
)

# ---------------- INPUT TYPE ----------------

option = st.radio(
    "Choose Input Type",
    ["Image Upload", "Video Upload"]
)

# ---------------- IMAGE DETECTION ----------------

if option == "Image Upload":

    st.markdown('<div class="upload-box">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:

        image = Image.open(uploaded_file)

        results = model(image)

        annotated_frame = results[0].plot()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖼 Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🧠 Detection Result")
            st.image(annotated_frame, use_container_width=True)

        st.markdown(
            '<div class="success-box">✅ Detection Completed Successfully!</div>',
            unsafe_allow_html=True
        )

# ---------------- VIDEO DETECTION ----------------

elif option == "Video Upload":

    st.markdown('<div class="upload-box">', unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "📤 Upload Video",
        type=["mp4", "mov", "avi"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_video:

        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_video.name)

        stframe = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            results = model(frame)

            annotated_frame = results[0].plot()

            annotated_frame = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            stframe.image(
                annotated_frame,
                channels="RGB",
                use_container_width=True
            )

        cap.release()

        os.unlink(temp_video.name)

        st.markdown(
            '<div class="success-box">✅ Video Processing Completed!</div>',
            unsafe_allow_html=True
        )

# ---------------- FOOTER ----------------

st.markdown(
    "<hr><center>Made with ❤️ using YOLOv8 + Streamlit</center>",
    unsafe_allow_html=True
)