import os
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image


# ==============================
# 1. Project paths and settings
# ==============================

MODEL_PATH = "models/cats_vs_dogs_best.keras"
OUTPUT_DIR = "sample_predictions"

IMAGE_SIZE = (224, 224)
NUM_SAMPLES = 10

random.seed(42)


# ==============================
# 2. Load trained model
# ==============================

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ==============================
# 3. Collect test images
# ==============================

cat_dir = "data/raw/dataset/PetImages/Cat"
dog_dir = "data/raw/dataset/PetImages/Dog"

image_paths = []

for filename in os.listdir(cat_dir):
    if filename.lower().endswith(".jpg"):
        image_paths.append((os.path.join(cat_dir, filename), "Cat"))

for filename in os.listdir(dog_dir):
    if filename.lower().endswith(".jpg"):
        image_paths.append((os.path.join(dog_dir, filename), "Dog"))


# ==============================
# 4. Select random samples
# ==============================

random.shuffle(image_paths)

selected_samples = image_paths[:NUM_SAMPLES]


# ==============================
# 5. Create output folder
# ==============================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# 6. Generate predictions
# ==============================

print("\nGenerating sample predictions...\n")

for index, (image_path, true_label) in enumerate(selected_samples, start=1):

    # Load original image
    image = Image.open(image_path).convert("RGB")

    # Resize image for model
    resized_image = image.resize(IMAGE_SIZE)

    # Convert image to NumPy array
    image_array = np.array(resized_image, dtype=np.float32)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Model prediction
    prediction = model.predict(image_array, verbose=0)[0][0]

    # Convert probability into label
    if prediction >= 0.5:
        predicted_label = "Dog"
        confidence = prediction
    else:
        predicted_label = "Cat"
        confidence = 1 - prediction

    # Check whether prediction is correct
    result = "Correct" if predicted_label == true_label else "Incorrect"

    # Print result
    print(
        f"{index}. "
        f"Predicted: {predicted_label} | "
        f"Actual: {true_label} | "
        f"Confidence: {confidence:.2%} | "
        f"{result}"
    )

    # ==============================
    # 7. Create visualization
    # ==============================

    plt.figure(figsize=(6, 6))

    plt.imshow(image)
    plt.axis("off")

    plt.title(
        f"Predicted: {predicted_label} ({confidence:.2%})\n"
        f"Ground Truth: {true_label}"
    )

    # Save image
    output_path = os.path.join(
        OUTPUT_DIR,
        f"sample_{index}_{predicted_label}_actual_{true_label}.png"
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
        dpi=150
    )

    plt.close()


# ==============================
# 8. Completion message
# ==============================

print("\n====================================")
print("SAMPLE INFERENCE COMPLETE!")
print("====================================")
print(f"Generated samples : {NUM_SAMPLES}")
print(f"Saved in          : {OUTPUT_DIR}/")