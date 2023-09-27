#!/usr/bin/env python3

import cv2
import tensorflow as tf
import numpy as np

# Load the pre-trained cat face cascade classifier
cat_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")

# Load a pre-trained model for cat color classification
model = tf.keras.models.load_model("cat_color_model.h5")

# Define a function to classify the color of a cat
def classify_cat_color(image):
    # Resize the image to the input size expected by the model (224x224)
    image = cv2.resize(image, (224, 224))
    
    # Convert the image to RGB color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to [0, 1]
    image = image / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Predict the color
    predictions = model.predict(image)
    color_index = np.argmax(predictions)
    if color_index == 0:
        return "Black"
    elif color_index == 1:
        return "White"
    else:
        return "Unknown"

# Initialize the webcam or use a video file
cap = cv2.VideoCapture(0)  # 0 for the default webcam, or specify a video file path

while True:
    # Read a frame from the video source
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cat faces in the frame
    cat_faces = cat_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50)
    )

    # Classify the color of each detected cat face
    for x, y, w, h in cat_faces:
        cat_face = frame[y : y + h, x : x + w]
        color = classify_cat_color(cat_face)

        # Draw a rectangle and label with the color
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    # Display the frame with cat faces and their colors
    cv2.imshow("Cat Color Detection", frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video source and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
