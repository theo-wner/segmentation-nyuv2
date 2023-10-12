import argparse

"""
Defines the Hyperparameters as command line arguments
"""

parser = argparse.ArgumentParser(description='Parser')

parser.add_argument('--num_classes', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=6e-5)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--devices', type=int, default=0)
parser.add_argument('--cpu', type=bool, default=False)
parser.add_argument('--checkpoint', type=str, default=None)

args = parser.parse_args()

IGNORE_INDEX = 255
NUMBER_TRAIN_IMAGES = 795
NUMBER_TEST_IMAGES = 654
NUM_CLASSES = args.num_classes
BATCH_SIZE = args.batch_size
NUM_WORKERS =  args.num_workers
NUM_EPOCHS = args.num_epochs
LEARNING_RATE = args.lr
WEIGHT_DECAY = args.weight_decay
DEVICES = [args.devices]
CPU_USAGE = args.cpu
CHECKPOINT = args.checkpoint

# Command for training 13 classes with checkpoint:
# python train.py --num_classes 13 --num_epochs 2000 --devices 1 --path_to_checkpoint ./tb_logs/nyuv2_13_classes/version_0/checkpoints/epoch=999-step=50000.ckpt

# Command for tensorboard:
# tensorboard --logdir tb_logs/ --bind_all