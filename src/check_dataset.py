from pathlib import Path
from PIL import Image

# Dataset folders
dataset_path = Path("data/raw/dataset/PetImages")

classes = ["Cat", "Dog"]

total_images = 0
valid_images = 0
corrupted_images = []

for class_name in classes:
    class_path = dataset_path / class_name

    image_files = list(class_path.glob("*.jpg"))

    print(f"\nChecking {class_name} images...")
    print(f"Found: {len(image_files)} images")

    for image_path in image_files:
        total_images += 1

        try:
            with Image.open(image_path) as image:
                image.verify()

            valid_images += 1

        except Exception as error:
            corrupted_images.append((str(image_path), str(error)))


print("\n" + "=" * 50)
print("DATASET INTEGRITY REPORT")
print("=" * 50)

print(f"Total images checked : {total_images}")
print(f"Valid images         : {valid_images}")
print(f"Corrupted images     : {len(corrupted_images)}")

if corrupted_images:
    print("\nCorrupted images:")
    
    for image_path, error in corrupted_images:
        print(f"- {image_path}")
        print(f"  Error: {error}")

else:
    print("\nNo corrupted JPG images found.")