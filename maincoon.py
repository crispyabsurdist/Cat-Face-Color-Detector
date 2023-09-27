#!/usr/bin/env python3

import cv2
import tensorflow as tf
import numpy as np

# Constants
CASCADE_PATH = "haarcascade_frontalcatface.xml"
MODEL_PATH = "cat_color_model.h5"
COLORS = ["Black", "White", "Unknown"]
VIDEO_SOURCE = 0  # 0 for default webcam
RECTANGLE_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (0, 255, 0)
FONT_THICKNESS = 2
MIN_FACE_SIZE = (50, 50)
SCALE_FACTOR = 1.3
MIN_NEIGHBORS = 5

# Load cascade and model
cat_cascade = cv2.CascadeClassifier(CASCADE_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

def process_image(image):
    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    return image

def classify_cat_color(image):
    processed_image = process_image(image)
    predictions = model.predict(processed_image)
    color_index = np.argmax(predictions)
    return COLORS[color_index]

# Initialize video capture
cap = cv2.VideoCapture(VIDEO_SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cat_faces = cat_cascade.detectMultiScale(gray, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS, minSize=MIN_FACE_SIZE)

    for x, y, w, h in cat_faces:
        cat_face = frame[y : y + h, x : x + w]
        color = classify_cat_color(cat_face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), RECTANGLE_COLOR, 2)
        cv2.putText(frame, color, (x, y - 10), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)

    cv2.imshow("CatFace Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
