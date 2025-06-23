import os
import tkinter as tk
from tkinter import filedialog
import cv2
import csv

# --- Setup ---
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

# --- Ask user for image folder ---
root = tk.Tk()
root.withdraw()

folder = filedialog.askdirectory(title="Select Folder with Images")
if not folder:
    print("No folder selected. Exiting.")
    exit()

image_files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(SUPPORTED_FORMATS)]

if not image_files:
    print("No image files found in the selected folder.")
    exit()

# --- Annotation function ---
def annotate_image(image_path):
    img = cv2.imread(image_path)
    clone = img.copy()
    coords = []

    def draw_points():
        annotated = clone.copy()
        for i, (x, y) in enumerate(coords):
            cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(annotated, str(i+1), (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return annotated

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            coords.append((x, y))
            updated = draw_points()
            cv2.imshow("Annotate", updated)

    cv2.imshow("Annotate", img)
    cv2.setMouseCallback("Annotate", click_event)

    print(f"\nAnnotating image: {os.path.basename(image_path)}")
    print("  - Left-click to add point")
    print("  - Press 'u' to undo last point")
    print("  - Press any other key to finish image")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('u'):
            if coords:
                coords.pop()
                cv2.imshow("Annotate", draw_points())
                print("  ↩️  Undo last point")
            else:
                print("  ⚠️  No points to undo")
        else:
            break

    cv2.destroyAllWindows()
    return coords

# --- Main Loop ---
for img_file in image_files:
    full_path = os.path.join(folder, img_file)
    clicks = annotate_image(full_path)

    if not clicks:
        print(f"No points added for {img_file}, skipping CSV.")
        continue

    csv_name = os.path.splitext(img_file)[0] + ".csv"
    csv_path = os.path.join(folder, csv_name)
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for x, y in clicks:
            writer.writerow([x, y])
    print(f"Saved {len(clicks)} points to {csv_name}")

print("✅ Annotation complete.")
