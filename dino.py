import torch
from torchvision import transforms
from PIL import Image
import os

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(device)

transform = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


image_name = "right-2025-05-21-18-55-05.png"
image = Image.open(f'./screenshots/{image_name}').convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

# Get features with registers
with torch.no_grad():
    output = model(image_tensor)