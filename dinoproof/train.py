from dinoproof.classifier import TerminationClassifier
import time
import argparse

parser = argparse.ArgumentParser(description="dino-proofreading")
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=4)
args = parser.parse_args()

output_dir = "weights/" + time.strftime("%Y-%m-%d-%H-%M-%S")
input_dir = args.input_dir

classifier = TerminationClassifier()
classifier.run_train(input_dir=args.input_dir, output_dir=output_dir, num_epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size)

