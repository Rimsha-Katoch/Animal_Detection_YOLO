import streamlit as st
import cv2
import numpy as np
import tempfile
import time

from streamlit_webrtc import webrtc_streamer
from av import VideoFrame

from detector import detect_animals

st.set_page_config(page_title="Animal Detection System",page_icon="🐾",layout="centered")

st.sidebar.title("About")

st.sidebar.write(
    """
    AI-powered Animal Detection System
    using YOLOv8 and OpenCV.

    Features:
    - Image Detection
    - Webcam Detection
    - Video Detection
    - Download Results
    """
)

st.title("🐾 Animal Detection System")

st.write("Upload images/videos or use webcam for real-time animal detection")

# IMAGE DETECTION SECTION

st.header("🖼 Image Detection")

uploaded_file = st.file_uploader("Choose an image",type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Convert uploaded image to numpy array
    file_bytes = np.asarray( bytearray(uploaded_file.read()),dtype=np.uint8)

    # Decode image
    image = cv2.imdecode(file_bytes, 1)

    # Spinner
    with st.spinner("Detecting animals..."):

        # Detect animals
        detected_image = detect_animals(image)

        # Convert BGR to RGB
        detected_image = cv2.cvtColor(detected_image,cv2.COLOR_BGR2RGB)

    # Display image
    st.image(
        detected_image,
        caption="Detection Result",
        use_container_width=True
    )

    st.success("Detection completed successfully!")

    # Convert image for download
    download_image = cv2.cvtColor(detected_image,cv2.COLOR_RGB2BGR)

    # Encode image
    _, buffer = cv2.imencode(".jpg",download_image)

    # Download button
    st.download_button(
        label="Download Detected Image",
        data=buffer.tobytes(),
        file_name="detected_image.jpg",
        mime="image/jpeg"
    )

# WEBCAM DETECTION SECTION

st.header("📷 Live Webcam Detection")

# Global variable for screenshot
save_frame = False

# Webcam callback function
def video_frame_callback(frame):

    global save_frame

    # Convert VideoFrame to numpy array
    img = frame.to_ndarray(format="bgr24")

    # Detect animals
    img = detect_animals(img)

    # Save screenshot only when button clicked
    if save_frame:

        filename = f"outputs/webcam_{int(time.time())}.jpg"

        cv2.imwrite(filename, img)

        save_frame = False

    # Return processed frame
    return VideoFrame.from_ndarray(
        img,
        format="bgr24"
    )

# Screenshot button
if st.button("📸 Save Webcam Screenshot"):
    save_frame = True

# Start webcam
webrtc_streamer(
    key="animal-detection",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)

# VIDEO DETECTION SECTION

st.header("📹 Video Upload Detection")

uploaded_video = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"],
    key="video"
)

if uploaded_video is not None:

    # Create temporary file
    temp_video = tempfile.NamedTemporaryFile(
        delete=False
    )

    # Save uploaded video
    temp_video.write(uploaded_video.read())

    st.success("Video uploaded successfully!")

    # Open video
    cap = cv2.VideoCapture(temp_video.name)

    # Check video
    if not cap.isOpened():

        st.error("Error opening video!")

    else:

        st.success("Video ready for detection!")

        # Placeholder for frames
        frame_placeholder = st.empty()

        # Process frames
        while True:

            ret, frame = cap.read()

            # Stop when video ends
            if not ret:
                break

            # Detect animals
            frame = detect_animals(frame)

            # Convert BGR to RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Display frame
            frame_placeholder.image(
                frame,
                channels="RGB",
                use_container_width=True
            )

        cap.release()

        st.success("Video processing completed!")