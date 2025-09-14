import os
import pandas as pd

# Input and output folders
input_folder = "./screenshots/TEST_DATASETS/test_set_1_augmented_392"
output_folder = "./screenshots/TEST_DATASETS/test_set_1_augmented_512"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Scaling factor
scale = 512 / 392

# Iterate over all files in input_folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".csv"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Read CSV
        df = pd.read_csv(input_path)

        # Multiply numeric values by scale and round to nearest int
        df = df.applymap(
            lambda x: int(round(x * scale)) if isinstance(x, (int, float)) else x
        )

        # Save to new folder with the same name
        df.to_csv(output_path, index=False)

        print(f"Processed {filename} → {output_path}")
