#!/bin/bash
#SBATCH --job-name=dino_v3_test     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of GPUs per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=kp0374@princeton.edu

module purge
module load anacondapy/2024.02
conda activate dino-proofreading
pip install -e .

python dinoproof/inference.py --input_dir "./screenshots/TEST_DATASETS/test_set_1_augmented_512" --weights_dir "./weights/2025-11-15-13-55-33/model_epoch_15.pth"