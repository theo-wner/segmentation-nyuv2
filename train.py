import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import config
from dataset import NYUv2DataModule
from model import DeepLab

if __name__ == '__main__':

    # Initialize the logger
    logger = TensorBoardLogger('tb_logs/', name='nyuv2_v0')
    
    # Initialize the model
    model = DeepLab()

     # Initialize the data module
    data_module = NYUv2DataModule(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

    # Initialize the trainer
    trainer = pl.Trainer(logger=logger, max_epochs=config.NUM_EPOCHS, accelerator='gpu', devices=config.DEVICES)

    # Train the model
    trainer.fit(model, data_module)


