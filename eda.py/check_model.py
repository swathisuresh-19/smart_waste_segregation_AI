import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model(
    "waste_classifier.keras"
)

print("Model loaded successfully!")

# Class names
classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# Ask for image
image_path = input(
    "\nEnter the full path of a waste image: "
).strip().strip('"')

# Load image
image = Image.open(image_path).convert("RGB")

# Resize
image = image.resize((224, 224))

# Convert to NumPy
image_array = np.array(image)

# IMPORTANT:
# Do NOT divide by 255 here because
# your saved model already contains preprocessing.

image_array = np.expand_dims(
    image_array,
    axis=0
)

# Prediction
prediction = model.predict(
    image_array,
    verbose=0
)[0]

print("\nPrediction probabilities:")

for class_name, probability in zip(classes, prediction):
    print(
        f"{class_name}: {probability * 100:.2f}%"
    )

predicted_index = np.argmax(prediction)

print(
    "\nPredicted class:",
    classes[predicted_index]
)

print(
    f"Confidence: {prediction[predicted_index] * 100:.2f}%"
)