import torch
import torch.nn as nn
import numpy as np
import os
from torchvision import transforms
from torch.nn.functional import interpolate
from PIL import Image

class TerminationClassifier(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.size = 392  # Size of the input images

        self.model = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=3, padding=1),  # Local context
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)  # Final output logits
        )

    def forward(self, feature_grid):
        return self.model(feature_grid).squeeze(1)

    def extract_points(self, csv_path):
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        return data

    def generate_heatmap(self, points):

        sigma = 2

        heatmap = np.zeros((self.size, self.size), dtype=np.float32)

        for x,y in points:
            x = int(round(x))
            y = int(round(y))
            if 0 <= x < self.size and 0 <= y < self.size:
                # Create a grid of coordinates
                xv, yv = np.meshgrid(np.arange(self.size), np.arange(self.size))

                # Calculate squared distance from the point
                dist_sq = (xv - x) ** 2 + (yv - y) ** 2

                # Compute Gaussian
                gauss = np.exp(-dist_sq / (2 * sigma ** 2))

                # Combine with existing heatmap (supports multiple points)
                heatmap = np.maximum(heatmap, gauss)

        return torch.tensor(heatmap)

    def load_dataset(self, image_folder, dino_model, device):
        transform = transforms.ToTensor()

        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg'))]

        image_tensors = []
        heatmap_tensors = []

        transform = transforms.Compose([
            transforms.Resize((self.size, self.size)), # If not already resized
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            
            image = Image.open(image_path).convert("RGB")
            img_tensor = transform(image).to(device)

            csv_path = image_path.rsplit('.', 1)[0] + ".csv"

            points = self.extract_points(csv_path)

            heatmap_tensor = self.generate_heatmap(points)

            heatmap_tensors.append(heatmap_tensor)

            image_tensors.append(img_tensor)

        return image_tensors, heatmap_tensors

    def train_model(self, dino_model, device, output_dir, input_dir, num_epochs=10, learning_rate=0.001, batch_size=4):
        self.to(device)
        dino_model.eval().to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        # Obtain training data
        images_tensor, heatmaps_tensor =  self.load_dataset(image_folder=input_dir, dino_model=dino_model, device=device)

        images_tensor = torch.stack(images_tensor).to(device)
        heatmaps_tensor = torch.stack(heatmaps_tensor).to(device)
        n = images_tensor.shape[0]

        os.makedirs(output_dir, exist_ok=True)

        for epoch in range(num_epochs):
            total_loss = 0.0

            print(f"Epoch {epoch + 1}/{num_epochs}, start training")

            perm = torch.randperm(n)
            images_shuffled = images_tensor[perm]
            heatmaps_shuffled = heatmaps_tensor[perm]

            for i in range(0, n, batch_size):
                batch_images = images_shuffled[i:i + batch_size]
                batch_heatmaps = heatmaps_shuffled[i:i + batch_size]

                with torch.no_grad():
                    output = dino_model.forward_features(batch_images)
                    raw_feature_grid = output["x_norm_patchtokens"]
                    B, _, C = raw_feature_grid.shape  # B: batch size, N: number of patches, C: feature dimension
                    raw_feature_grid = raw_feature_grid.reshape(B, 28, 28, C)  # [Batch size, height, width, feature dimension]

                    # compute per-point feature using bilinear interpolation
                    interpolated_feature_grid = interpolate(raw_feature_grid.permute(0, 3, 1, 2),  # Rearrange [Batch size, feature_dim, patch_h, patch_w]
                                                            size=(self.size, self.size),
                                                            mode='bilinear')
                    batch_features = interpolated_feature_grid

                # get features from images
                batch_features = batch_features.to(device)
                batch_heatmaps = batch_heatmaps.to(device)

                logits = self.forward(batch_features)
                loss = criterion(logits, batch_heatmaps)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                print(f"  Batch {i // batch_size + 1} - Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / (n // batch_size)
            print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}, end training")

            save_path = os.path.join(output_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(self.state_dict(), save_path)
            print(f"Weights saved!")

    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    test_file = "right-2025-06-13-22-47-31_90"

    classifier = TerminationClassifier(feature_dim=384)  # Assuming DINO features are 384-dimensional

    points = classifier.extract_points(f"./screenshots/raw_1_augmented/{test_file}.csv")

    heatmap = classifier.generate_heatmap(points)

    plt.figure(figsize=(8,6))
    plt.imshow(np.array(Image.open(f"./screenshots/raw_1_augmented/{test_file}.png").convert("RGB")))
    heatmap = heatmap.detach().cpu().squeeze().numpy()
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    plt.show()