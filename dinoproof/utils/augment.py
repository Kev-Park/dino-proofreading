import os
import cv2
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path


def rotate_points(points, rotation_matrix):
    """
    Applies the OpenCV affine rotation matrix to a set of (x, y) points.

    Args:
        points (ndarray): Array of shape (N, 2) with x, y coordinates.
        rotation_matrix (ndarray): 2x3 affine transformation matrix.

    Returns:
        rotated_points (ndarray): Array of rotated points with shape (N, 2).
    """
    # Convert to homogeneous coordinates (add ones for translation handling)
    ones = np.ones((points.shape[0], 1))
    homogenous_points = np.hstack([points, ones])  # Shape (N, 3)

    # Apply affine transformation
    rotated_points = homogenous_points @ rotation_matrix.T  # Shape (N, 2)

    return rotated_points


def augment_folder(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    image_files = sorted(glob(str(input_folder / "*.*g")))  # Matches png, jpg, jpeg

    for img_path in image_files:
        img_path = Path(img_path)
        base_name = img_path.stem  # filename without extension
        csv_path = input_folder / f"{base_name}.csv"

        if not csv_path.exists():
            print(f"⚠️ Warning: CSV not found for {img_path.name}, skipping.")
            continue

        # Load image and CSV
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]

        df = pd.read_csv(csv_path)
        if not set(df.columns) >= {"x", "y"}:
            print(f"⚠️ Warning: CSV {csv_path.name} missing 'x' or 'y' columns, skipping.")
            continue

        points = df[['x', 'y']].values.astype(np.float32)

        for angle in [0, 90, 180, 270]:
            # Compute rotation matrix
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

            # Rotate image
            rotated_img = cv2.warpAffine(image, M, (w, h))

            # Rotate points using the same matrix
            rotated_points = rotate_points(points, M)

            # Save rotated image
            out_img_name = f"{base_name}_{angle}.png"
            cv2.imwrite(str(output_folder / out_img_name), rotated_img)

            # Save rotated CSV
            out_csv_name = f"{base_name}_{angle}.csv"
            rotated_df = pd.DataFrame(rotated_points, columns=["x", "y"])
            rotated_df.to_csv(str(output_folder / out_csv_name), index=False)

            print(f"✅ Saved {out_img_name} and {out_csv_name}")

    print("🎉 Augmentation complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Augment image+CSV pairs by rotation.")
    parser.add_argument("--input_folder", type=str, help="Input folder containing images and CSVs.")
    parser.add_argument("--output_folder", type=str, help="Output folder for augmented files.")

    args = parser.parse_args()

    augment_folder(args.input_folder, args.output_folder)
