# -*- coding: utf-8 -*-
"""License plate detection for local images.

This script works without Google Colab. If no image path is passed, it creates a
small demo image so the detection workflow can be verified in a local environment.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def create_demo_image(output_path: Path) -> np.ndarray:
    """Create a simple synthetic car image with a visible license plate."""
    image = np.full((600, 1000, 3), 180, dtype=np.uint8)

    cv2.rectangle(image, (80, 180), (920, 500), (90, 90, 90), -1)
    cv2.rectangle(image, (120, 220), (860, 420), (60, 80, 220), -1)
    cv2.rectangle(image, (250, 145), (640, 220), (180, 180, 180), -1)

    cv2.rectangle(image, (280, 300), (720, 390), (255, 255, 255), -1)
    cv2.rectangle(image, (280, 300), (720, 390), (0, 0, 0), 3)
    cv2.putText(
        image,
        "TN 09 AB 1234",
        (310, 355),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(output_path), image)
    return image


def display(image: np.ndarray, title: str = "") -> None:
    """Display an image using matplotlib."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(rgb_image)
    plt.axis("off")
    plt.title(title)
    plt.show()


def detect_plate(image: np.ndarray):
    """Detect a license plate in a vehicle image."""
    plate_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)

        if len(approx) == 4:
            x, y, width, height = cv2.boundingRect(approx)
            aspect_ratio = width / float(height)

            if 2.0 <= aspect_ratio <= 6.0 and width > 60:
                plate = image[y : y + height, x : x + width]
                cv2.rectangle(
                    plate_image,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    3,
                )
                return plate_image, plate

    return plate_image, None


def resolve_image_path(arg_path: str | None) -> Path:
    """Return an image path: provided path if given, otherwise use a demo file."""
    if arg_path:
        return Path(arg_path).expanduser().resolve()

    for candidate in [
        Path.cwd() / "demo_license_plate.jpg",
        Path.cwd() / "sample.jpg",
        Path.cwd() / "car.jpg",
    ]:
        if candidate.exists():
            return candidate

    demo_path = Path.cwd() / "demo_license_plate.jpg"
    create_demo_image(demo_path)
    return demo_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect a vehicle license plate in an image.")
    parser.add_argument("--image", help="Path to the input image file.")
    parser.add_argument("--show", action="store_true", help="Display the output image window.")
    args = parser.parse_args()

    image_path = resolve_image_path(args.image)
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Unable to read image at: {image_path}")

    output_image, plate = detect_plate(image)

    output_file = Path.cwd() / "detected_license_plate.jpg"
    cv2.imwrite(str(output_file), output_image)

    if plate is not None:
        plate_file = Path.cwd() / "extracted_license_plate.jpg"
        cv2.imwrite(str(plate_file), plate)
        print(f"Image processed successfully: {image_path}")
        print(f"Detected plate saved to: {output_file}")
        print(f"Extracted plate saved to: {plate_file}")
    else:
        print(f"No license plate detected in {image_path}. Output saved to {output_file}.")

    if args.show:
        display(output_image, "Detected License Plate")
        if plate is not None:
            display(plate, "Extracted License Plate")


if __name__ == "__main__":
    main()

