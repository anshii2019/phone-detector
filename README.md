# 📱 AI Phone Usage Detection System

## 📌 Overview

AI Phone Usage Detection System is a real-time computer vision application that detects mobile phone usage through a webcam using YOLOv8. The system monitors live video streams, identifies phones and people in the frame, and generates alerts when phone usage is detected.

The project also includes a simple dashboard for tracking phone usage violations.

---

## 🚀 Features

* Real-time phone detection using webcam
* Person and mobile phone recognition
* Live alert display when phone usage is detected
* Streamlit-based web interface
* Violation tracking dashboard
* Optimized YOLOv8 inference for real-time performance
* User-friendly interface

---

## 🛠️ Tech Stack

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* Streamlit
* Pandas
* Streamlit-WebRTC

---

## 📂 Project Structure

```text
phone_detector/
│
├── app.py              # Main Streamlit application
├── main.py             # Initial testing script
├── yolov8n.pt          # YOLOv8 model
├── requirements.txt    # Project dependencies
├── alert/              # Alert-related files
└── model/              # Model-related files
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/phone-detector.git
cd phone-detector
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Open your browser and visit:

```text
http://localhost:8501
```

---

## 🧠 How It Works

1. Webcam captures live video frames.
2. YOLOv8 processes each frame.
3. The model detects:

   * Person
   * Cell Phone
4. If both are detected:

   * Alert is displayed.
   * Violation is recorded.
5. Dashboard visualizes phone usage activity.

---

## 🎯 Use Cases

* Classroom Monitoring
* Employee Productivity Tracking
* Exam Hall Monitoring
* Workplace Compliance Monitoring
* Smart Surveillance Systems

---

## 📈 Future Improvements

* Custom-trained phone detection model
* Face recognition integration
* Violation report export (PDF/Excel)
* Cloud deployment
* Email/SMS alerts
* Database integration

---

## 👩‍💻 Author

**Anshika Arya**

AI/ML Engineer

---

## ⭐ Acknowledgements

* Ultralytics YOLOv8
* Streamlit
* OpenCV Community
* Python Open Source Ecosystem

```
```
