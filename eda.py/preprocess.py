import tensorflow as tf

# -----------------------------
# Configuration
# -----------------------------

DATASET_PATH = "dataset-resized"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# -----------------------------
# Load 70% Training Dataset
# -----------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.30,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# -----------------------------
# Load 30% Temporary Dataset
# -----------------------------

temp_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.30,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# -----------------------------
# Split Temporary Dataset
# 50% Validation + 50% Test
# -----------------------------

temp_batches = tf.data.experimental.cardinality(temp_dataset)

print("\nTemporary batches:", temp_batches.numpy())

validation_batches = temp_batches.numpy() // 2

validation_dataset = temp_dataset.take(validation_batches)
test_dataset = temp_dataset.skip(validation_batches)

# -----------------------------
# Class Names
# -----------------------------

class_names = train_dataset.class_names

print("\nClasses:")

for index, name in enumerate(class_names):
    print(index, ":", name)

# -----------------------------
# Dataset Information
# -----------------------------

print("\nNumber of classes:", len(class_names))
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)

# -----------------------------
# Check Training Batch
# -----------------------------

for images, labels in train_dataset.take(1):

    print("\nTraining image batch shape:", images.shape)
    print("Training label batch shape:", labels.shape)

# -----------------------------
# Performance Optimization
# -----------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)

print("\nDataset splitting completed successfully!")

print("\nExpected split:")
print("Training:   approximately 70%")
print("Validation: approximately 15%")
print("Testing:    approximately 15%")