from pathlib import Path
from PIL import Image


# Dataset folder
dataset_path = Path("data/raw/dataset/PetImages")

classes = ["Cat", "Dog"]

total_images = 0
valid_images = 0
corrupted_images = []


# Cat aur Dog dono folders check karna
for class_name in classes:

    class_path = dataset_path / class_name

    image_files = list(class_path.glob("*.jpg"))

    print(f"\nChecking {class_name} images...")
    print(f"Found: {len(image_files)} images")

    for image_path in image_files:

        total_images += 1

        try:
            # Image structure verify karna
            with Image.open(image_path) as image:
                image.verify()

            # Image ko dobara open karke
            # actual pixel data load karna
            with Image.open(image_path) as image:
                image.convert("RGB").load()

            valid_images += 1

        except Exception as error:

            corrupted_images.append(
                (str(image_path), str(error))
            )


# Final report
print("\n" + "=" * 50)
print("DATASET INTEGRITY REPORT")
print("=" * 50)

print(f"Total images checked : {total_images}")
print(f"Valid images         : {valid_images}")
print(f"Corrupted images     : {len(corrupted_images)}")


# Agar corrupted images milti hain
if corrupted_images:

    print("\nCorrupted images:")

    for image_path, error in corrupted_images:

        print(f"- {image_path}")
        print(f"  Error: {error}")

else:

    print("\nNo corrupted JPG images found.")