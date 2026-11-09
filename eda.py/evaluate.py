import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# Configuration
# ==========================================

DATASET_PATH = "dataset-resized"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

MODEL_PATH = "waste_classifier.keras"

# ==========================================
# Load Trained Model
# ==========================================

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ==========================================
# Load Dataset
# ==========================================

temp_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.30,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ==========================================
# Create Test Dataset
# ==========================================

total_batches = tf.data.experimental.cardinality(temp_dataset).numpy()

validation_batches = total_batches // 2

test_dataset = temp_dataset.skip(validation_batches)

class_names = temp_dataset.class_names

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, ":", name)

# ==========================================
# Get Actual Labels
# ==========================================

y_true = []
y_pred = []

print("\nMaking predictions...")

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

# Convert to NumPy arrays

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ==========================================
# Classification Report
# ==========================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================\n")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================\n")

print(cm)

# ==========================================
# Display Confusion Matrix
# ==========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    xticks_rotation=45
)

plt.title("Waste Classification - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved as:")
print("confusion_matrix.png")