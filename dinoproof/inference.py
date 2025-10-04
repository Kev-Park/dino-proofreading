from dinoproof.classifier import TerminationClassifier

import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from torch.nn.functional import interpolate
from PIL import Image
import argparse

# Parse CMD-line args
parser = argparse.ArgumentParser(description="dino-proofreading")
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--weights_dir", type=str, required=True)
args = parser.parse_args()

# Load classifier
classifier = TerminationClassifier()
classifier.eval()
state_dict = torch.load(args.weights_dir, map_location=classifier.device)

# Run inference
all_images, _ = classifier.load_image(image_path=args.input_dir, generate_heatmap=False, normalize=False)


# Save results
output_dir = "results/" + time.strftime("%Y-%m-%d-%H-%M-%S")
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "model_weights.txt"), "w") as f:
    f.write(args.weights_dir)

# Operate over batches of 4 images
for i in range(0, len(all_images), 4):
    images = all_images[i:i+4]

    images = torch.stack(images).to(classifier.device)
    features = classifier.embed(images)
    heatmaps = classifier.forward(features)

    # Get model heatmap
    with torch.no_grad():
        model_heatmap = classifier.forward(features)
        # Apply sigmoid to get probabilities
        model_heatmap = torch.sigmoid(model_heatmap)
    model_heatmap = model_heatmap.cpu().squeeze()
    images = images.cpu()

    # Get real heatmap
    # real_heatmap = classifier.generate_heatmap(classifier.extract_points(f"./screenshots/{dataset_name}/{test_file}.csv"))

    print("Saving results",flush=True)
    for j in range(len(model_heatmap)):

        plt.figure(figsize=(10, 5),layout='constrained')
        ax1 = plt.subplot(1, 2, 1)
        img = images[j].permute(1, 2, 0)


        ax1.imshow(img)
        #ax1.imshow(real_heatmap, alpha=0.5, cmap="jet")
        ax1.set_title("Original Image")
        ax1.set_xticks(np.linspace(0, img.shape[1], 5))
        ax1.set_yticks(np.linspace(0, img.shape[0], 5))

        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(img)
        img_heat = ax2.imshow(model_heatmap[j], alpha=0.5, cmap="jet")
        ax2.set_title("Model Predicted Heatmap")
        ax2.set_xticks(np.linspace(0, img.shape[1], 5))
        ax2.set_yticks(np.linspace(0, img.shape[0], 5))

        cbar1 = plt.colorbar(img_heat)
        cbar1.set_label("Heatmap Intensity")

        plt.suptitle(f"Test {4*i+j}", fontsize=16)
        plt.savefig(os.path.join(output_dir, f"result-{4*i+ j}.png"))