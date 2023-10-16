import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import config
from dataset import NYUv2DataModule
from model import MiT

"""
Trains the model
"""

if __name__ == '__main__':
    # Initialize the logger
    name = 'nyuv2_' + str(config.NUM_CLASSES) + '_classes'
    logger = TensorBoardLogger('tb_logs/', name=name)
    
    # Initialize the model
    model = MiT()

    # Initialize the data module
    data_module = NYUv2DataModule(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

    # Initialize the trainer
    if config.CPU_USAGE:
        trainer = pl.Trainer(logger=logger, max_epochs=config.NUM_EPOCHS, accelerator='cpu')
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=config.NUM_EPOCHS, accelerator='gpu', devices=config.DEVICES)

    # Train the model
    if config.CHECKPOINT is not None:
        trainer.fit(model, data_module, ckpt_path=config.CHECKPOINT)
    trainer.fit(model, data_module)


# Command for training 13 classes with checkpoint:
# python train.py --num_classes 13 --num_epochs 2000 --devices 1 --path_to_checkpoint ./tb_logs/nyuv2_13_classes/version_0/checkpoints/epoch=999-step=50000.ckpt

# Command for tensorboard:
# tensorboard --logdir tb_logs/ --bind_all