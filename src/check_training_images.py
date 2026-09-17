from pathlib import Path
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent.parent

CAT_DIR = PROJECT_DIR / "data" / "raw" / "dataset" / "PetImages" / "Cat"
DOG_DIR = PROJECT_DIR / "data" / "raw" / "dataset" / "PetImages" / "Dog"


def check_folder(folder, class_name):

    print()
    print("=" * 60)
    print(f"CHECKING {class_name} IMAGES")
    print("=" * 60)

    files = list(folder.glob("*.jpg"))

    bad_files = []

    for index, path in enumerate(files, start=1):

        try:
            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                image.convert("RGB").load()

        except Exception as error:

            bad_files.append(
                (path, str(error))
            )

        if index % 1000 == 0:
            print(f"Checked {index} images...")

    print()
    print(f"Total checked : {len(files)}")
    print(f"Bad images   : {len(bad_files)}")

    if bad_files:

        print()
        print("PROBLEMATIC FILES:")

        for path, error in bad_files:
            print()
            print("File :", path)
            print("Error:", error)

    else:
        print("No problematic images found.")


check_folder(
    CAT_DIR,
    "CAT"
)

check_folder(
    DOG_DIR,
    "DOG"
)