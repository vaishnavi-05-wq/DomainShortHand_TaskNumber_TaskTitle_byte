from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("data/raw/dataset/PetImages")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

CLASS_NAMES = ["Cat", "Dog"]


# ============================================================
# STEP 1: COLLECT IMAGE PATHS AND LABELS
# ============================================================

image_paths = []
labels = []

for label, class_name in enumerate(CLASS_NAMES):

    class_path = DATASET_DIR / class_name

    image_files = sorted(class_path.glob("*.jpg"))

    for image_path in image_files:
        image_paths.append(str(image_path))
        labels.append(label)


image_paths = np.array(image_paths)
labels = np.array(labels)


# ============================================================
# STEP 2: DATASET INFORMATION
# ============================================================

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Total images : {len(image_paths)}")
print(f"Cat images   : {np.sum(labels == 0)}")
print(f"Dog images   : {np.sum(labels == 1)}")


# ============================================================
# STEP 3: TRAIN + TEST SPLIT
# ============================================================

train_val_paths, test_paths, train_val_labels, test_labels = (
    train_test_split(
        image_paths,
        labels,
        test_size=0.15,
        random_state=SEED,
        stratify=labels
    )
)


# ============================================================
# STEP 4: TRAIN + VALIDATION SPLIT
# ============================================================

train_paths, validation_paths, train_labels, validation_labels = (
    train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=0.17647,
        random_state=SEED,
        stratify=train_val_labels
    )
)


# ============================================================
# STEP 5: DISPLAY SPLIT INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET SPLIT")
print("=" * 60)

print(f"Training images   : {len(train_paths)}")
print(f"Validation images : {len(validation_paths)}")
print(f"Test images       : {len(test_paths)}")


print("\nClass distribution:")

print(
    f"Train      → Cat: {np.sum(train_labels == 0)}, "
    f"Dog: {np.sum(train_labels == 1)}"
)

print(
    f"Validation → Cat: {np.sum(validation_labels == 0)}, "
    f"Dog: {np.sum(validation_labels == 1)}"
)

print(
    f"Test       → Cat: {np.sum(test_labels == 0)}, "
    f"Dog: {np.sum(test_labels == 1)}"
)


# ============================================================
# STEP 6: IMAGE LOADING FUNCTION
# ============================================================

def load_image(image_path, label):

    # Convert TensorFlow string tensor to Python string
    image_path = image_path.numpy().decode("utf-8")

    # Open image using Pillow
    image = Image.open(image_path)

    # Convert every image to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize(IMAGE_SIZE)

    # Convert image to NumPy array
    image = np.array(image, dtype=np.float32)

    # Normalize pixels from 0–255 to 0–1
    image = image / 255.0

    return image, np.float32(label)


# ============================================================
# STEP 7: TENSORFLOW WRAPPER
# ============================================================

def load_image_tf(image_path, label):

    image, label = tf.py_function(
        func=load_image,
        inp=[image_path, label],
        Tout=[tf.float32, tf.float32]
    )

    # Tell TensorFlow the expected shapes
    image.set_shape(
        [IMAGE_SIZE[0], IMAGE_SIZE[1], 3]
    )

    label.set_shape([])

    return image, label


# ============================================================
# STEP 8: CREATE DATASET
# ============================================================

def create_dataset(paths, labels, shuffle=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        load_image_tf,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:

        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# STEP 9: CREATE TRAIN / VALIDATION / TEST DATASETS
# ============================================================

train_dataset = create_dataset(
    train_paths,
    train_labels,
    shuffle=True
)

validation_dataset = create_dataset(
    validation_paths,
    validation_labels
)

test_dataset = create_dataset(
    test_paths,
    test_labels
)


# ============================================================
# STEP 10: VERIFY ONE BATCH
# ============================================================

for images, labels_batch in train_dataset.take(1):

    print("\n" + "=" * 60)
    print("SAMPLE BATCH INFORMATION")
    print("=" * 60)

    print(f"Images shape : {images.shape}")
    print(f"Labels shape : {labels_batch.shape}")
    print(f"Image dtype  : {images.dtype}")
    print(f"Label dtype  : {labels_batch.dtype}")

    print(
        f"Pixel minimum : {tf.reduce_min(images).numpy():.4f}"
    )

    print(
        f"Pixel maximum : {tf.reduce_max(images).numpy():.4f}"
    )

    break


# ============================================================
# STEP 11: FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("DATASET PIPELINE READY!")
print("=" * 60)

print("✓ Dataset collected")
print("✓ Stratified train/validation/test split")
print("✓ Images converted to RGB")
print("✓ Images resized to 224 × 224")
print("✓ Pixel values normalized to 0–1")
print("✓ TensorFlow datasets created")