import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.utils.class_weight import compute_class_weight


# ==========================================
# Configuration
# ==========================================

DATASET_PATH = "dataset-resized"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

NUM_CLASSES = 6

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 5


# ==========================================
# Load Training Dataset
# ==========================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.30,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ==========================================
# Load Validation/Test Dataset
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
# Split Validation and Test
# ==========================================

total_batches = tf.data.experimental.cardinality(
    temp_dataset
).numpy()

validation_batches = total_batches // 2

validation_dataset = temp_dataset.take(
    validation_batches
)

test_dataset = temp_dataset.skip(
    validation_batches
)


# ==========================================
# Class Names
# ==========================================

class_names = train_dataset.class_names

print("\nClasses:")

for i, name in enumerate(class_names):
    print(i, ":", name)


# ==========================================
# Calculate Class Weights
# ==========================================

all_labels = []

for _, labels in train_dataset:
    all_labels.extend(labels.numpy())

all_labels = np.array(all_labels)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=all_labels
)

class_weights = {
    i: class_weights_array[i]
    for i in range(NUM_CLASSES)
}

print("\nClass weights:")

for i, name in enumerate(class_names):
    print(
        name,
        ":",
        round(class_weights[i], 3)
    )


# ==========================================
# Data Augmentation
# ==========================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])


# ==========================================
# MobileNetV2
# ==========================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# ==========================================
# Build Model
# ==========================================

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)


# ==========================================
# Compile
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# Callbacks
# ==========================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_waste_classifier.keras",
    monitor="val_accuracy",
    save_best_only=True
)


# ==========================================
# Initial Training
# ==========================================

print("\n==========================================")
print("INITIAL TRAINING")
print("==========================================\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# ==========================================
# Fine-Tuning
# ==========================================

print("\n==========================================")
print("FINE-TUNING MOBILENETV2")
print("==========================================\n")

base_model.trainable = True

# Freeze the first 100 layers
for layer in base_model.layers[:100]:
    layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


history_fine = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# ==========================================
# Load Best Model
# ==========================================

model = tf.keras.models.load_model(
    "best_waste_classifier.keras"
)


# ==========================================
# Evaluate
# ==========================================

print("\n==========================================")
print("FINAL TEST EVALUATION")
print("==========================================\n")

test_loss, test_accuracy = model.evaluate(
    test_dataset
)

print("\nTest Loss:", test_loss)

print(
    "Test Accuracy:",
    test_accuracy
)


# ==========================================
# Save Final Model
# ==========================================

model.save(
    "waste_classifier_improved.keras"
)

print("\n==========================================")
print("MODEL SAVED")
print("==========================================")

print(
    "waste_classifier_improved.keras"
)