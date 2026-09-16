# ============================================================
# AVIP Task 1 - Cats vs Dogs
# Model Evaluation
# ============================================================

import os
import random
import numpy as np
import tensorflow as tf

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_STATE = 42

DATASET_DIR = "data/raw/dataset/PetImages"
MODEL_PATH = "models/cats_vs_dogs_best.keras"

RESULTS_DIR = "results"
CONFUSION_MATRIX_PATH = os.path.join(
    RESULTS_DIR, "confusion_matrix.txt"
)

# ------------------------------------------------------------
# 2. Collect image paths and labels
# ------------------------------------------------------------

image_paths = []
labels = []

class_names = ["Cat", "Dog"]

for label, class_name in enumerate(class_names):

    class_dir = os.path.join(DATASET_DIR, class_name)

    for filename in os.listdir(class_dir):

        if filename.lower().endswith((".jpg", ".jpeg", ".png")):

            image_paths.append(
                os.path.join(class_dir, filename)
            )

            labels.append(label)

image_paths = np.array(image_paths)
labels = np.array(labels)

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Total images : {len(image_paths)}")
print(f"Cat images   : {np.sum(labels == 0)}")
print(f"Dog images   : {np.sum(labels == 1)}")

# ------------------------------------------------------------
# 3. Recreate the same stratified split
# ------------------------------------------------------------

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=RANDOM_STATE
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    stratify=temp_labels,
    random_state=RANDOM_STATE
)

print("\n" + "=" * 60)
print("TEST SET INFORMATION")
print("=" * 60)

print(f"Test images : {len(test_paths)}")
print(f"Test Cats   : {np.sum(test_labels == 0)}")
print(f"Test Dogs   : {np.sum(test_labels == 1)}")

# ------------------------------------------------------------
# 4. Image loading function
# ------------------------------------------------------------

def load_image(path, label):

    def load_with_pillow(path_bytes):

        path_string = path_bytes.numpy().decode("utf-8")

        with Image.open(path_string) as image:

            image = image.convert("RGB")
            image = image.resize(IMAGE_SIZE)

            image_array = np.asarray(
                image,
                dtype=np.float32
            )

        return image_array / 255.0

    image = tf.py_function(
        func=load_with_pillow,
        inp=[path],
        Tout=tf.float32
    )

    image.set_shape(
        [IMAGE_SIZE[0], IMAGE_SIZE[1], 3]
    )

    label = tf.cast(label, tf.float32)

    return image, label


# ------------------------------------------------------------
# 5. Create test dataset
# ------------------------------------------------------------

test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)

test_dataset = test_dataset.map(
    load_image,
    num_parallel_calls=1
)

test_dataset = test_dataset.batch(BATCH_SIZE)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)

# ------------------------------------------------------------
# 6. Load trained model
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LOADING TRAINED MODEL")
print("=" * 60)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print(f"Model loaded successfully:")
print(MODEL_PATH)

# ------------------------------------------------------------
# 7. Generate predictions
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("RUNNING MODEL ON TEST DATA")
print("=" * 60)

probabilities = model.predict(
    test_dataset,
    verbose=1
)

probabilities = probabilities.ravel()

predicted_labels = (
    probabilities >= 0.5
).astype(int)

true_labels = test_labels.astype(int)

# ------------------------------------------------------------
# 8. Calculate evaluation metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision = precision_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

recall = recall_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

# ------------------------------------------------------------
# 9. Display results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        zero_division=0
    )
)

# ------------------------------------------------------------
# 10. Save confusion matrix and metrics
# ------------------------------------------------------------

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

metrics_path = os.path.join(
    RESULTS_DIR,
    "evaluation_metrics.txt"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("AVIP TASK 1 - CATS VS DOGS\n")
    file.write("MODEL EVALUATION RESULTS\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Test images : {len(test_paths)}\n")
    file.write(f"Test Cats   : {np.sum(test_labels == 0)}\n")
    file.write(f"Test Dogs   : {np.sum(test_labels == 1)}\n\n")

    file.write(f"Accuracy  : {accuracy:.4f}\n")
    file.write(f"Precision : {precision:.4f}\n")
    file.write(f"Recall    : {recall:.4f}\n")
    file.write(f"F1-score  : {f1:.4f}\n\n")

    file.write("Confusion Matrix:\n")
    file.write(
        np.array2string(cm)
    )

    file.write("\n\nClassification Report:\n")

    file.write(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=class_names,
            zero_division=0
        )
    )

with open(
    CONFUSION_MATRIX_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Confusion Matrix\n"
    )

    file.write(
        "Rows = Actual, Columns = Predicted\n\n"
    )

    file.write(
        "             Predicted Cat   Predicted Dog\n"
    )

    file.write(
        f"Actual Cat       {cm[0][0]:5d}          {cm[0][1]:5d}\n"
    )

    file.write(
        f"Actual Dog       {cm[1][0]:5d}          {cm[1][1]:5d}\n"
    )

print("\n" + "=" * 60)
print("RESULTS SAVED")
print("=" * 60)

print(f"Metrics file          : {metrics_path}")
print(f"Confusion matrix file : {CONFUSION_MATRIX_PATH}")

print("\nEVALUATION COMPLETE!")