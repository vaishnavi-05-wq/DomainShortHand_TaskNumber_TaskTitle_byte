from pathlib import Path
from collections import Counter

from PIL import Image


# Dataset location
dataset_path = Path("data/raw/dataset/PetImages")

# Classes
classes = ["Cat", "Dog"]


# Store image dimensions
dimensions = []

# Count images for each class
class_counts = {}


# Check every class
for class_name in classes:

    class_path = dataset_path / class_name

    image_files = list(class_path.glob("*.jpg"))

    class_counts[class_name] = len(image_files)

    print(f"Analyzing {class_name} images...")

    for image_path in image_files:

        try:
            with Image.open(image_path) as image:

                width, height = image.size

                dimensions.append((width, height))

        except Exception:
            print(f"Could not read: {image_path}")


# Separate widths and heights
widths = [width for width, height in dimensions]
heights = [height for width, height in dimensions]


# Most common dimensions
dimension_counts = Counter(dimensions)


print("\n" + "=" * 55)
print("IMAGE DIMENSION ANALYSIS")
print("=" * 55)

print(f"Cat images : {class_counts['Cat']}")
print(f"Dog images : {class_counts['Dog']}")
print(f"Total images analyzed : {len(dimensions)}")

print(f"\nMinimum width  : {min(widths)}")
print(f"Maximum width  : {max(widths)}")

print(f"Minimum height : {min(heights)}")
print(f"Maximum height : {max(heights)}")


print("\nMost common image dimensions:")

for (width, height), count in dimension_counts.most_common(10):

    print(f"{width} × {height} : {count} images")