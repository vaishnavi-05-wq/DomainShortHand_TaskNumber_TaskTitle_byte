from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image


# Dataset location
dataset_path = Path("data/raw/dataset/PetImages")

# Classes
classes = ["Cat", "Dog"]

# Number of images to display from each class
samples_per_class = 6

# Make random selection reproducible
random.seed(42)


# Create a figure
fig, axes = plt.subplots(
    2,
    samples_per_class,
    figsize=(15, 7)
)


# Process each class
for row, class_name in enumerate(classes):

    class_path = dataset_path / class_name

    # Get all JPG images
    image_files = list(class_path.glob("*.jpg"))

    # Randomly select images
    selected_images = random.sample(
        image_files,
        samples_per_class
    )

    # Display selected images
    for column, image_path in enumerate(selected_images):

        with Image.open(image_path) as image:

            # Get image dimensions
            width, height = image.size

            # Display image
            axes[row, column].imshow(image)

            # Display label and dimensions
            axes[row, column].set_title(
                f"{class_name}\n{width} × {height}"
            )

            axes[row, column].axis("off")


# Add overall title
fig.suptitle(
    "Cats vs Dogs Dataset - Sample Images",
    fontsize=16
)

# Adjust layout
plt.tight_layout()

# Display the figure
plt.show()