from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("data/raw/dataset/PetImages")

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

CLASS_NAMES = ["Cat", "Dog"]

EPOCHS = 5


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


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


print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Total images : {len(image_paths)}")
print(f"Cat images   : {np.sum(labels == 0)}")
print(f"Dog images   : {np.sum(labels == 1)}")


# ============================================================
# STEP 2: TRAIN / VALIDATION / TEST SPLIT
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


train_paths, validation_paths, train_labels, validation_labels = (
    train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=0.17647,
        random_state=SEED,
        stratify=train_val_labels
    )
)


print("\n" + "=" * 60)
print("DATASET SPLIT")
print("=" * 60)

print(f"Training images   : {len(train_paths)}")
print(f"Validation images : {len(validation_paths)}")
print(f"Test images       : {len(test_paths)}")


# ============================================================
# STEP 3: IMAGE LOADING
# ============================================================

def load_image(image_path, label):

    image_path = image_path.numpy().decode("utf-8")

    image = Image.open(image_path)

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image = np.array(
        image,
        dtype=np.float32
    )

    image = image / 255.0

    return image, np.float32(label)


def load_image_tf(image_path, label):

    image, label = tf.py_function(
        func=load_image,
        inp=[image_path, label],
        Tout=[tf.float32, tf.float32]
    )

    image.set_shape(
        [IMAGE_SIZE[0], IMAGE_SIZE[1], 3]
    )

    label.set_shape([])

    return image, label


# ============================================================
# STEP 4: CREATE TF.DATA DATASETS
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
            seed=SEED
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


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
# STEP 5: DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.1
        ),

        tf.keras.layers.RandomZoom(
            0.1
        ),
    ],
    name="data_augmentation"
)


# ============================================================
# STEP 6: LOAD MOBILENETV2
# ============================================================

print("\n" + "=" * 60)
print("LOADING MOBILENETV2")
print("=" * 60)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),
    include_top=False,
    weights="imagenet"
)


# Freeze pretrained layers

base_model.trainable = False


print("MobileNetV2 loaded successfully.")
print("Pretrained layers frozen.")


# ============================================================
# STEP 7: BUILD CLASSIFICATION MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)


# Apply augmentation

x = data_augmentation(inputs)


# MobileNetV2 feature extraction

x = base_model(
    x,
    training=False
)


# Convert feature maps into one vector

x = tf.keras.layers.GlobalAveragePooling2D()(x)


# Reduce overfitting

x = tf.keras.layers.Dropout(
    0.2
)(x)


# Binary classification output

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid"
)(x)


model = tf.keras.Model(
    inputs,
    outputs,
    name="cats_vs_dogs_mobilenetv2"
)


# ============================================================
# STEP 8: COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# STEP 9: DISPLAY MODEL
# ============================================================

print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)

model.summary()


# ============================================================
# STEP 10: CALLBACKS
# ============================================================

best_model_path = (
    MODEL_DIR / "cats_vs_dogs_best.keras"
)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1
    )
]


# ============================================================
# STEP 11: TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ============================================================
# STEP 12: SAVE FINAL MODEL
# ============================================================

final_model_path = (
    MODEL_DIR / "cats_vs_dogs_final.keras"
)

model.save(
    final_model_path
)


print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(f"Best model  : {best_model_path}")
print(f"Final model : {final_model_path}")