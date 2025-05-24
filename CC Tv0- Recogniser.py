import cv2
import torch
from ultralytics import YOLO
import face_recognition
import os
import numpy as np
import time
import smtplib
from email.message import EmailMessage

# Email alert function
def send_alert_email(image_path):
    msg = EmailMessage()
    msg["Subject"] = "🚨 Unknown Person Detected"
    msg["From"] = "your_email@gmail.com"
    msg["To"] = "receiver_email@gmail.com"
    msg.set_content("Unknown person detected. See attached image.")

    with open(image_path, "rb") as f:
        img_data = f.read()
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename="unknown.jpg")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login("your_email@gmail.com", "your_app_password")
        smtp.send_message(msg)

# Load YOLOv8 model
yolo_model = YOLO('yolov8n.pt')  # Use 'yolov8s.pt' or better for accuracy

# Load known faces
known_face_encodings = []
known_face_names = []

known_faces_dir = "known_faces"
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(os.path.splitext(filename)[0])

# Start video capture
cap = cv2.VideoCapture(0)  # Replace 0 with your IP stream if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster face recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Face detection
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            print("🚨 New person detected at", time.strftime("%Y-%m-%d %H:%M:%S"))
            
            # Save face snapshot
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            unknown_face = frame[top*2:bottom*2, left*2:right*2]
            filename = f"unknown_{timestamp}.jpg"
            cv2.imwrite(filename, unknown_face)

            # Send email alert
            send_alert_email(filename)

        # Draw face box
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # YOLOv8 Object detection
    results = yolo_model(frame, verbose=False)[0]
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{yolo_model.names[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("AI Surveillance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
