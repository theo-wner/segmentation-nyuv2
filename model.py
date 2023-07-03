import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

class FPN(pl.LightningModule):
    def __init__(self):
        super.__init__()

        # Initialize the model
        self.model = smp.FPN(
            encoder_name='se_resnext50_32x4d',
            encoder_weights='imagenet', 
            classes=15,
            activation='softmax'
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = F.cross_entropy(scores, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = F.cross_entropy(scores, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = F.cross_entropy(scores, y)
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        return scores
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)