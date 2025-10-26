import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Load model and haarcascade
model = load_model('model_file_30epochs.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

# Streamlit page settings
st.set_page_config(page_title="Face Emotion Detection", layout="centered")
st.title("😊 Real-Time Face Emotion Detection")
st.write("Upload an image or use your webcam to detect emotions!")

# File uploader
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])

def detect_emotion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)
    for (x, y, w, h) in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(img, labels_dict[label], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img

# If user uploads an image
if uploaded_file is not None:
    img = np.array(Image.open(uploaded_file))
    st.image(img, caption="Uploaded Image", use_column_width=True)
    result_img = detect_emotion(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Detected Emotion", use_column_width=True)

# Webcam live detection
st.write("---")
st.subheader("🎥 Try Real-Time Webcam Detection")
run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)
while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Could not access webcam.")
        break
    frame = cv2.flip(frame, 1)
    detected_frame = detect_emotion(frame)
    FRAME_WINDOW.image(cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB))

camera.release()
