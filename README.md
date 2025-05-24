# AI-CCtv-surveilence-and-face-detection
An AI-powered smart CCTV surveillance system using Python, OpenCV, YOLOv8, and DeepSort. Detects faces, tracks objects, identifies unusual activity, and sends real-time alerts. Ideal for smart homes, offices, and public safety systems.
# 🛡️ AI CCTV Surveillance System

An intelligent real-time surveillance system powered by AI. This project uses Python, OpenCV, YOLOv8, and DeepSort to monitor CCTV footage, detect and track people or objects, recognize faces, and alert users about unusual activities.

## 🔍 Features

- ✅ Real-time object detection using YOLOv8
- ✅ Multi-object tracking with DeepSort
- ✅ Face detection and recognition
- ✅ Unusual activity detection (e.g., loitering, crowd formation)
- ✅ Real-time alerts via notifications (optional)
- ✅ License Plate Recognition (optional)
- ✅ Lightweight and cross-platform

## 📌 Tech Stack

- **Python**
- **OpenCV**
- **Ultralytics YOLOv8**
- **DeepSort**
- **NumPy**
- **Tkinter** / **Streamlit** (optional dashboard)

## 📸 How It Works

1. Load the CCTV feed (from IP cam or webcam)
2. Run YOLOv8 to detect objects (e.g., persons, vehicles)
3. Track objects with DeepSort to assign consistent IDs
4. Apply logic to detect suspicious or unusual activity
5. Optionally, recognize faces or license plates
6. Trigger alerts and log the incident

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/ai-cctv-surveillance.git
cd ai-cctv-surveillance
