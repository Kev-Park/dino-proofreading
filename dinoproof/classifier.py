import torch
import torch.nn as nn
import numpy as np
import os
from torchvision import transforms
from torch.nn.functional import interpolate
import torch.nn.functional as F
from PIL import Image

class TerminationClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = 16 # Default
        self.embedding_dim = 768 # 384, 768, 1024, 1280, or 4096
        self.image_size = 512#512  # Size of the input images
        self.dino = None

        # Nonlinear
        # self.model = nn.Sequential(
        #     nn.Conv2d(feature_dim, 64, kernel_size=3, padding=1),  # 384 -> 64
        #     nn.ReLU(),
        #     nn.Conv2d(64, 32, kernel_size=3, padding=1), # 64 -> 32
        #     nn.ReLU(),
        #     nn.Conv2d(32, 1, kernel_size=1) # 32 -> 1
        # )

        # Linear
        self.model = nn.Sequential(
            nn.Conv2d(self.embedding_dim, 128, kernel_size=3, padding=1, bias=True), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=True), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=True)
        )
        
        # Decoder
        # self.model = nn.Sequential(
        #     nn.Conv2d(768, 256, 3, padding=1), nn.ReLU(),
        #     nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),  # 32 -> 64
        #     nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),   # 64 -> 128
        #     nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),    # 128 -> 256
        #     nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),    # 256 -> 512
        #     nn.Conv2d(16, 1, 1)  # final 1-channel heatmap
        # )

        self.to(self.device)
        # Load DINOv3 B16 (76M)
        #self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(self.device)
        self.dino = torch.hub.load(repo_or_dir='facebookresearch/dinov3', model='dinov3_vitb16', weights='dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth').eval().to(self.device)        

    def extract_points(self, csv_path):
        """
        Extract (x, y) points from a CSV file.
        """

        try:
            data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        except:
            # Return empty array if file not found or empty
            data = np.empty((0, 2))
        return data

    def generate_heatmap(self, points):
        """
        Generate a heatmap from a list of (x, y) points using Gaussians.
        """

        sigma = 5

        heatmap = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        for x,y in points:
            x = int(round(x))
            y = int(round(y))
            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                # Create a grid of coordinates
                xv, yv = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))

                # Calculate squared distance from the point
                dist_sq = (xv - x) ** 2 + (yv - y) ** 2

                # Compute Gaussian
                gauss = np.exp(-dist_sq / (2 * sigma ** 2))

                # Combine with existing heatmap (supports multiple points)
                heatmap = np.maximum(heatmap, gauss)
                
                # Filter out values smaller than threshold
                threshold = 0.01
                heatmap[heatmap < threshold] = 0

        return torch.tensor(heatmap)

    def load_image(self, image_path, generate_heatmap = False, normalize = True):
        """
        Load a single image or a directory of images and convert them to tensors.

        generate_heatmap: If True, will attempt to fetch ground truth heatmaps (assuming labeled data).
        """

        # Establish image transformations for model size + normalization
        transform = transforms.ToTensor()
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)), # If not already resized
            transforms.ToTensor(),
        ])
        normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # If generating heatmap
        image_tensors = []
        heatmap_tensors = None
        if generate_heatmap:
            heatmap_tensors = []

        # If directory of images to embed is given
        if os.path.isdir(image_path):
            image_paths = [os.path.join(image_path,f) for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg'))]
        else:
            image_paths = [image_path]

        # Transform image, load as tensors, optionally generate heatmap
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            img_tensor = transform(image).to(self.device)
            if normalize:
                img_tensor = normalize_transform(img_tensor)
            image_tensors.append(img_tensor)

            if generate_heatmap:
                csv_path = image_path.rsplit('.', 1)[0] + ".csv"
                points = self.extract_points(csv_path)
                heatmap_tensor = self.generate_heatmap(points)
                heatmap_tensors.append(heatmap_tensor)
        return image_tensors, heatmap_tensors
        
    def embed(self, image_tensors_batch):
        """
        Get DINOv3 features from a batch of image tensors.
        """

        with torch.no_grad():
            output = self.dino.forward_features(image_tensors_batch)
        raw_feature_grid = output["x_norm_patchtokens"]
        B, _, C = raw_feature_grid.shape  # B: batch size, N: number of patches, C: feature dimension
        patch_count=int(self.image_size/self.patch_size)
        raw_feature_grid = raw_feature_grid.reshape(B, patch_count, patch_count, C)  # [Batch size, height, width, feature dimension]

        #return raw_feature_grid.permute(0, 3, 1, 2)  # Rearrange to [Batch size, feature_dim, patch_h, patch_w]

        #compute per-point feature using bilinear interpolation
        interpolated_feature_grid = interpolate(raw_feature_grid.permute(0, 3, 1, 2),  # Rearrange [Batch size, feature_dim, patch_h, patch_w]
                                                size=(self.image_size, self.image_size),
                                                mode='bilinear')
        batch_features = interpolated_feature_grid
        return batch_features

    def forward(self, feature_grid):
        return self.model(feature_grid).squeeze(1)

    def load_dataset(self, image_path):
        """
        Load dataset of images and corresponding heatmaps.
        """

        image_tensors, heatmap_tensors = self.load_image(image_path=image_path, generate_heatmap=True)
        return torch.stack(image_tensors).to(self.device), torch.stack(heatmap_tensors).to(self.device)

    def run_train(self, output_dir, input_dir, num_epochs=10, learning_rate=0.0001, batch_size=4):

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        #pos_weight = torch.tensor([200.0], device=self.device)  
        #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        #criterion = nn.MSELoss()
        criterion = nn.BCEWithLogitsLoss()

        # Obtain training data
        images_tensor, heatmaps_tensor =  self.load_dataset(image_path=input_dir)

        # Get number of samples
        n = images_tensor.shape[0]

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Train
        for epoch in range(num_epochs):
            total_loss = 0.0

            print(f"Epoch {epoch + 1}/{num_epochs}, start training")

            perm = torch.randperm(n)
            images_shuffled = images_tensor[perm]
            heatmaps_shuffled = heatmaps_tensor[perm]

            for i in range(0, n, batch_size):
                batch_images = images_shuffled[i:i + batch_size]
                batch_heatmaps = heatmaps_shuffled[i:i + batch_size]

                batch_features = self.embed(batch_images)

                # Get features from images
                batch_features = batch_features.to(self.device)
                batch_heatmaps = batch_heatmaps.to(self.device)

                logits = self.forward(batch_features)
                loss = criterion(logits, batch_heatmaps)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                print(f"  Batch {i // batch_size + 1} - Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / (n // batch_size)
            print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}, end training")

            if epoch+1 % 10 == 0:
                save_path = os.path.join(output_dir, f"model_epoch_{epoch + 1}.pth")
                torch.save(self.state_dict(), save_path)
                print(f"Weights saved!")

    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    test_file = "right-2025-06-13-22-47-31_0"

    classifier = TerminationClassifier()

    image, heatmap = classifier.load_image(generate_heatmap=True, image_path="screenshots/misc/test")

    plt.figure(figsize=(8,6))
    plt.imshow(np.array(image[0].cpu().permute(1,2,0)))
    #heatmap = heatmap.detach().cpu().squeeze().numpy()
    plt.imshow(heatmap[0].cpu().squeeze(), alpha=0.5, cmap='jet')
    plt.show()



    # points = classifier.extract_points(f"./screenshots/TRAIN_DATASETS/raw_1_false_positive_augmented_512/{test_file}.csv")

    # heatmap = classifier.generate_heatmap(points)

    # plt.figure(figsize=(8,6))
    # plt.imshow(np.array(Image.open(f"./screenshots/TRAIN_DATASETS/raw_1_false_positive_augmented_512/{test_file}.png").convert("RGB").resize((classifier.image_size, classifier.image_size))))
    # heatmap = heatmap.detach().cpu().squeeze().numpy()
    # plt.imshow(heatmap, alpha=0.5, cmap='jet')
    # plt.show()