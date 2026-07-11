# Face and Object Recognition Application  
### Using Python, OpenCV, and MobileNet SSD

This project is a Python-based application that performs **real-time face detection** and **object recognition** using OpenCV. The system can recognize common objects such as **school buses, cars, cell phones, bottles, chairs, and people**, along with **frontal human faces**. The application uses a combination of **Haar Cascade classifiers** for face detection and **MobileNet SSD** for object recognition.

---

## 📌 Features

- Real-time webcam detection  
- Face detection using Haar Cascade  
- Object recognition using MobileNet SSD (COCO classes)  
- Bounding boxes and confidence scores  
- Screenshot capture for documentation  
- Organized project structure for easy submission and grading  

---

## 📁 Project Structure

FaceObjectRecognition/
│
├── models/
│   ├── MobileNetSSD_deploy.prototxt
│   ├── MobileNetSSD_deploy.caffemodel
│   └── haarcascade_frontalface_default.xml
│
├── src/
│   ├── app.py
│   └── utils.py
│
├── screenshots/
│   ├── face_detection.png
│   ├── bus_detection.png
│   ├── cellphone_detection.png
│   └── other_objects.png
│
├── requirements.txt
└── README.md


---

## 🔧 Installation

1. Install Python 3.10+  
2. Install dependencies:

```bash
pip install -r requirements.txt 


Download the MobileNet SSD model files:

MobileNetSSD_deploy.prototxt

MobileNetSSD_deploy.caffemodel

Place them inside the models/ folder.


## ▶️ Running the Application

Run the main script:

bash
python src/app.py
Controls:

Press q to quit

## 🧠 How It Works
Face Detection

The application uses OpenCV’s Haar Cascade classifier to detect frontal faces.
It converts each frame to grayscale and identifies face regions using pattern matching.

Object Recognition

The system uses the MobileNet SSD deep learning model trained on the COCO dataset.
Each frame is converted into a blob and passed through the network to detect objects such as:

* Person

* Bus

* Car

* Bottle

* Chair

* TV Monitor

* Cell Phone (detected as “cell phone” or “mobile device” depending on model)

Bounding boxes and confidence scores are drawn on the frame.

## 📝 Requirements
Python 3.10+

OpenCV

NumPy

MobileNet SSD model files

Haar Cascade XML file

## 👩‍💻 Author
Tatiana  
Master’s-level Computer Science Student
Lorain, Ohio

## 📄 License
This project is for academic use only.