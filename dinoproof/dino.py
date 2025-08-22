import torch
from torch.nn.functional import interpolate
from torchvision import transforms
from PIL import Image
import os

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').eval().to(device)

transform = transforms.Compose([
    transforms.Resize((392, 392)), # If not already resized
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_name = "right-2025-05-21-18-55-05.png"
image = Image.open(f'./screenshots/{image_name}').convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

# Get features with registers
with torch.no_grad():
    output = model.forward_features(image_tensor)
    raw_feature_grid = output["x_norm_patchtokens"]
    raw_feature_grid = raw_feature_grid.reshape(1, 28, 28, -1)  # float32 [num_cams, patch_h, patch_w, feature_dim]
    # compute per-point feature using bilinear interpolation
    interpolated_feature_grid = interpolate(raw_feature_grid.permute(0, 3, 1, 2),  # float32 [num_cams, feature_dim, patch_h, patch_w]
                                            size=(392, 392),
                                            mode='bilinear').permute(0, 2, 3, 1).squeeze(0)  # float32 [H, W, feature_dim]
    features_flat = interpolated_feature_grid.reshape(-1, interpolated_feature_grid.shape[-1])  # float32 [H*W, feature_dim]

    print(features_flat.shape)  # Should be [15488, 384] for 14x14 patches
    #print(output["x_norm_regtokens"].shape)
    
    
