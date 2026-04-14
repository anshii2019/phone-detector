import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import winsound
import datetime
import time
current_time = time.time()          
import os
os.makedirs("alert", exist_ok=True)

# Page config
st.set_page_config(page_title="AI Phone Detector", layout="centered")

# UI
st.title("📱 AI Phone Usage Detector")
st.markdown("### Detect phone usage in images & live camera")


# Load model
model = YOLO("model/yolov8n.pt")

# ================= IMAGE UPLOAD =================
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    results = model(image)
    detected_objects = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            detected_objects.append(label)

    # 🔍 DEBUG (optional)
    st.write("Detected Objects:", detected_objects)

    # 🚨 ALERT LOGIC (FINAL)
    if "person" in detected_objects and any(x in detected_objects for x in ["cell phone", "remote"]):
        st.error("🚨 Phone Usage Detected!")
    else:
        st.success("✅ No Phone Usage")

    # Show images
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", channels="BGR")

    with col2:
        annotated = results[0].plot()
        st.image(annotated, caption="Detection Result", channels="BGR")


# ================= LIVE CAMERA =================
st.markdown("---")
st.subheader("🎥 Live Camera Detection")


run = st.checkbox("Start Camera")
if "count" not in st.session_state:
    st.session_state.count = 0

if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = 0

st.write("📊 Total Violations:", st.session_state.count)

if run:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not working")
            break

        results = model(frame)
        detected_objects = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                detected_objects.append(label)

        annotated = results[0].plot()

        # 🚨 ALERT LOGIC (LIVE)
        if "person" in detected_objects and any(x in detected_objects for x in ["cell phone", "remote"]):
            #🔴 Text
            cv2.putText(annotated, "PHONE DETECTED!", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            
            # ⏱️ TIMESTAMP (YAHI ADD KARNA HAI)
            time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cv2.putText(annotated, time_str, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            # 🔊 SOUND
            current_time = time.time()
            if current_time - st.session_state.last_alert_time > 2:  # 5 seconds cooldown
                winsound.Beep(1000, 500)
                st.session_state.last_alert_time = current_time

            # 📸 SAVE IMAGE
            #cv2.imwrite(f"alert/alert_{st.session_state.count}.jpg", annotated)
            cv2.imwrite(f"alert/alert_{time_str.replace(':','-')}.jpg", annotated)

            # 📊 COUNT
            st.session_state.count += 1

            


        stframe.image(annotated, channels="BGR")

    cap.release()