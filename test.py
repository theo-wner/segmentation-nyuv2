import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import config
from dataset import NYUv2DataModule
from model import DeepLab
import torch

"""
Tests the model
"""

if __name__ == '__main__':
    # Initialize the model
    model = DeepLab()
    if config.CPU_USAGE:
        checkpoint = torch.load("epoch=1999-step=200000.ckpt", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = model.load_from_checkpoint("epoch=1999-step=200000.ckpt")

    # Initialize the data module
    data_module = NYUv2DataModule(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

    # Initialize the logger
    name = 'nyuv2_' + str(config.NUM_CLASSES) + '_classes_test'
    logger = TensorBoardLogger('tb_logs/', name=name)

    # Initialize the trainer
    if config.CPU_USAGE:
        trainer = pl.Trainer(logger=logger, accelerator='cpu')
    else:
        trainer = pl.Trainer(logger=logger, accelerator='gpu', devices=config.DEVICES)

    # Test the model
    trainer.test(model, data_module)

# Command for testing:
# python test.py --devices 1
# python test.py --cpu True

# Command for tensorboard:
# tensorboard --logdir tb_logs/

