#!/usr/bin/env python3

import os
from dotenv import load_dotenv
import tensorflow as tf

# Dataset paths 
load_dotenv()
TRAIN_PATH = os.getenv("TRAIN_PATH")
VALID_PATH = os.getenv("VALID_PATH")

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"{TRAIN_PATH} does not exist")

# Create data generators
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

train_gen = datagen.flow_from_directory(
    TRAIN_PATH, target_size=(64, 64), batch_size=32, class_mode="categorical"
)

valid_gen = datagen.flow_from_directory(
    VALID_PATH, target_size=(64, 64), batch_size=32, class_mode="categorical"
)

# Load the MobileNetV2 model, excluding top layers
base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False)

# Add custom layers
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
predictions = tf.keras.layers.Dense(2, activation="softmax")(x)  # Assuming you have 2 classes: Black and White

model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Freeze layers from the base model
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model
model.fit(train_gen, validation_data=valid_gen, epochs=10)

# Save the trained model
model.save("cat_color_model.h5")
