import tensorflow as tf
import numpy as np
import csv
import os
from datetime import datetime
from tensorflow.keras.utils import load_img, img_to_array

# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "waste_classifier.keras"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

CSV_FILE = "predictions.csv"

# ==========================================
# Load Model
# ==========================================

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ==========================================
# Ask for Image
# ==========================================

image_path = input("\nEnter image path: ")

# ==========================================
# Load Image
# ==========================================

image = load_img(
    image_path,
    target_size=IMAGE_SIZE
)

image_array = img_to_array(image)

image_array = np.expand_dims(
    image_array,
    axis=0
)

# ==========================================
# Prediction
# ==========================================

predictions = model.predict(
    image_array,
    verbose=0
)

predicted_index = np.argmax(predictions[0])

predicted_class = CLASS_NAMES[predicted_index]

confidence = predictions[0][predicted_index] * 100

# ==========================================
# Display Result
# ==========================================

print("\n===================================")
print("       WASTE CLASSIFICATION")
print("===================================")

print(
    "Predicted Class:",
    predicted_class
)

print(
    "Confidence:",
    f"{confidence:.2f}%"
)

print("===================================")

# ==========================================
# Save Prediction to CSV
# ==========================================

file_exists = os.path.isfile(CSV_FILE)

with open(
    CSV_FILE,
    mode="a",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    # Create column names if file doesn't exist
    if not file_exists:
        writer.writerow([
            "Image",
            "Predicted Class",
            "Confidence",
            "Date",
            "Time"
        ])

    now = datetime.now()

    writer.writerow([
        os.path.basename(image_path),
        predicted_class,
        round(confidence, 2),
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S")
    ])

print("\nPrediction saved to predictions.csv")
