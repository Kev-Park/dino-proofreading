from classifier import TerminationClassifier
import torch
import time

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

    output_dir = "weights/" + time.strftime("%Y-%m-%d-%H-%M-%S")
    input_dir = "./screenshots/raw_1_augmented"

    num_epochs = 10
    learning_rate = 0.001
    batch_size = 4

    classifier = TerminationClassifier(feature_dim=384)

    classifier.train_model(dino_model=model, device=device, input_dir=input_dir, output_dir=output_dir, num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size)

