from dinoproof.classifier import TerminationClassifier
import torch
import time

if __name__ == "__main__":
    output_dir = "weights/" + time.strftime("%Y-%m-%d-%H-%M-%S")
    input_dir = "./screenshots/raw_1_false_positive_augmented"

    num_epochs = 10
    learning_rate = 0.001
    batch_size = 4

    classifier = TerminationClassifier()

    classifier.train(input_dir=input_dir, output_dir=output_dir, num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size)

