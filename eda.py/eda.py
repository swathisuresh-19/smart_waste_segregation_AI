import os
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# Dataset Path
# -----------------------------

DATASET_PATH = "dataset-resized"

# Get class names
classes = sorted(
    [
        folder for folder in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, folder))
    ]
)

print("Waste Classes:")
for cls in classes:
    print("-", cls)

# -----------------------------
# Count Images
# -----------------------------

class_counts = {}

for cls in classes:

    class_path = os.path.join(DATASET_PATH, cls)

    images = [
        file for file in os.listdir(class_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    class_counts[cls] = len(images)

print("\nImage Count by Class:")

for cls, count in class_counts.items():
    print(f"{cls}: {count}")

total_images = sum(class_counts.values())

print(f"\nTotal Images: {total_images}")

# -----------------------------
# Combined EDA Figure
# -----------------------------

fig = plt.figure(figsize=(14, 10))

# -----------------------------
# Bar Chart
# -----------------------------

ax1 = plt.subplot(2, 1, 1)

ax1.bar(class_counts.keys(), class_counts.values())

ax1.set_title("Waste Dataset - Class Distribution")
ax1.set_xlabel("Waste Category")
ax1.set_ylabel("Number of Images")

# -----------------------------
# Sample Images
# -----------------------------

ax_images = []

for cls in classes:

    class_path = os.path.join(DATASET_PATH, cls)

    images = [
        file for file in os.listdir(class_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if images:

        image_path = os.path.join(class_path, images[0])

        image = Image.open(image_path)

        ax_images.append((cls, image))

# Create 6 image panels
for i, (cls, image) in enumerate(ax_images):

    ax = fig.add_subplot(2, 6, i + 7)

    ax.imshow(image)
    ax.set_title(cls.capitalize())
    ax.axis("off")

plt.suptitle(
    "Smart Waste Segregation Dataset - Exploratory Data Analysis",
    fontsize=16
)

plt.tight_layout()

plt.show()

# -----------------------------
# Image Dimensions
# -----------------------------

print("\nImage Dimensions:")

for cls in classes:

    class_path = os.path.join(DATASET_PATH, cls)

    images = [
        file for file in os.listdir(class_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if images:

        image_path = os.path.join(class_path, images[0])

        with Image.open(image_path) as img:

            print(f"{cls}: {img.size}")