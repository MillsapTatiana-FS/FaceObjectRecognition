import cv2
import numpy as np
import time
import os

# Paths to model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

prototxt_path = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
model_path = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")
face_cascade_path = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")

# Load object detection model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load face detection model
face_cascade = cv2.CascadeClassifier(face_cascade_path)
print("Cascade loaded:", not face_cascade.empty())

# Object classes for MobileNet SSD
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

# Create screenshots folder if missing
SCREENSHOT_DIR = os.path.join(BASE_DIR, "..", "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 's' to save a screenshot.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Draw object detections
    h, w = frame.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}",
                        (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w_f, h_f) in faces:
        cv2.rectangle(frame, (x, y), (x + w_f, y + h_f),
                    (255, 0, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 0), 2)
    
    print("Displaying frame...")

    cv2.imshow("Face + Object Recognition", frame)

    key = cv2.waitKey(1)
    
    key = cv2.waitKey(1) & 0xFF

    if key & 0xFF == ord('s'):
        filename = os.path.join(SCREENSHOT_DIR, f"screenshot_{int(time.time())}.png")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

    if key & 0xFF == ord('q'):
        break

