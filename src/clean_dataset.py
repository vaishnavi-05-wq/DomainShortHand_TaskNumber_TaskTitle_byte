from pathlib import Path
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("data/raw/dataset/PetImages")

CLASS_NAMES = ["Cat", "Dog"]


# ============================================================
# CHECK ALL IMAGES
# ============================================================

total_images = 0
valid_images = 0
bad_images = []


for class_name in CLASS_NAMES:

    class_path = DATASET_DIR / class_name

    image_files = sorted(class_path.glob("*.jpg"))

    print(f"\nChecking {class_name} images...")
    print(f"Found: {len(image_files)}")

    for image_path in image_files:

        total_images += 1

        try:

            # Open image
            with Image.open(image_path) as image:

                # Check basic image information
                image.verify()

            # Re-open because verify() closes/checks the file
            with Image.open(image_path) as image:

                # Fully load image data
                image.load()

                # Check number of channels
                channels = len(image.getbands())

                if channels not in (1, 3, 4):

                    raise ValueError(
                        f"Invalid number of channels: {channels}"
                    )

            valid_images += 1

        except Exception as error:

            bad_images.append(
                (image_path, str(error))
            )


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 60)
print("IMAGE QUALITY REPORT")
print("=" * 60)

print(f"Total images checked : {total_images}")
print(f"Valid images         : {valid_images}")
print(f"Bad images           : {len(bad_images)}")


# ============================================================
# DISPLAY BAD IMAGES
# ============================================================

if bad_images:

    print("\nBad images found:")

    for image_path, error in bad_images:

        print(f"\nFile   : {image_path}")
        print(f"Reason : {error}")


# ============================================================
# DELETE ONLY BAD IMAGES
# ============================================================

if bad_images:

    print("\nRemoving bad images...")

    for image_path, _ in bad_images:

        image_path.unlink()

        print(f"Removed: {image_path}")

else:

    print("\nNo bad images found.")


print("\nDataset cleaning completed.")