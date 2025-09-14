import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_points(folder, filename_base):
    """
    Visualizes termination points over the image.

    Args:
        folder (str or Path): Folder where the image and CSV are located.
        filename_base (str): Filename without extension (e.g., 'image1_90')
    """
    folder = Path(folder)

    img_path = folder / f"{filename_base}.png"
    csv_path = folder / f"{filename_base}.csv"

    if not img_path.exists() or not csv_path.exists():
        print(f"❌ File not found: {img_path} or {csv_path}")
        return

    # Load image
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for plotting

    # Load CSV
    df = pd.read_csv(csv_path)
    if not set(df.columns) >= {"x", "y"}:
        print(f"❌ CSV {csv_path} does not contain 'x' and 'y' columns.")
        return

    x_coords = df["x"].values
    y_coords = df["y"].values

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.scatter(x_coords, y_coords, c='red', s=30, marker='o', label='Termination Points')
    plt.title(f"Visualization for {filename_base}")
    plt.legend()
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize image with termination points overlaid.")
    parser.add_argument("--folder", type=str, help="Folder containing image and CSV.")
    parser.add_argument("--filename", type=str, help="Base filename without extension (e.g., 'image1_90')")

    args = parser.parse_args()

    visualize_points(args.folder, args.filename)

