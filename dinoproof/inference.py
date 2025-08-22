import torch
from dinoproof.classifier import TerminationClassifier
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.nn.functional import interpolate
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = TerminationClassifier(feature_dim=384)

dataset_name = "false_positive_augmented"
test_file = "right-2025-06-26-21-07-29_180"
weights_path = "linear_false_positive/model_epoch_10"

classifier.load_state_dict(torch.load(f"./weights/{weights_path}.pth"))
classifier.eval().to(device)

# Get features
image_path = f"./screenshots/{dataset_name}/{test_file}.png"
transform = transforms.Compose([
            transforms.Resize((classifier.size, classifier.size)), # If not already resized
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
image = Image.open(image_path).convert("RGB")
img_tensor = transform(image).to(device).unsqueeze(0)  # Add batch dimension

dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').eval().to(device)
with torch.no_grad():
    output = dino_model.forward_features(img_tensor)
    raw_feature_grid = output["x_norm_patchtokens"]
    B, _, C = raw_feature_grid.shape  # B: batch size, N: number of patches, C: feature dimension
    raw_feature_grid = raw_feature_grid.reshape(B, 28, 28, C)  # [Batch size, height, width, feature dimension]

    # compute per-point feature using bilinear interpolation
    interpolated_feature_grid = interpolate(raw_feature_grid.permute(0, 3, 1, 2),  # Rearrange [Batch size, feature_dim, patch_h, patch_w]
                                            size=(classifier.size, classifier.size),
                                            mode='bilinear')
    features = interpolated_feature_grid

# Get model heatmap
with torch.no_grad():
    model_heatmap = classifier.forward(features)

# Get real heatmap
#real_heatmap = classifier.generate_heatmap(classifier.extract_points(f"./screenshots/{dataset_name}/{test_file}.csv"))

# Visualize
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(np.array(Image.open(f"./screenshots/{dataset_name}/{test_file}.png").convert("RGB")))
#plt.imshow(real_heatmap, alpha=0.5, cmap="jet")
plt.title("Ground Truth Heatmap")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(np.array(Image.open(f"./screenshots/{dataset_name}/{test_file}.png").convert("RGB")))
plt.imshow(model_heatmap.cpu().squeeze(), alpha=0.5, cmap="jet")
plt.title("Model Predicted Heatmap")
plt.axis("off")

plt.suptitle(test_file, fontsize=16)
plt.show()