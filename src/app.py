import cv2
import numpy as np
import os

# -----------------------------
# Load Model Files
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

prototxt_path = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
model_path = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")
cascade_path = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")

# Load MobileNet SSD model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# COCO-style class labels for MobileNet SSD
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cascade_path)

# -----------------------------
# Start Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

print("Press 's' to save a screenshot.")
print("Press 'q' to quit the application.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to access webcam.")
        break

    (h, w) = frame.shape[:2]

    # -----------------------------
    # Object Detection (MobileNet SSD)
    # -----------------------------
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.40:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            # Only highlight key objects
            if label in ["person", "bus", "car", "bottle", "chair", "tvmonitor"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, text, (startX, startY - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # -----------------------------
    # Face Detection (Haar Cascade)
    # -----------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w_f, h_f) in faces:
        cv2.rectangle(frame, (x, y), (x + w_f, y + h_f), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # -----------------------------
    # Display Frame
    # -----------------------------
    cv2.imshow("Face and Object Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s"):
        screenshot_path = os.path.join(BASE_DIR, "..", "screenshots", "screenshot.png")
        cv2.imwrite(screenshot_path, frame)
        print(f"Screenshot saved to {screenshot_path}")

cap.release()
cv2.destroyAllWindows()
