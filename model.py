import torch
import torchmetrics
from torch import nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import ssl
import math
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import config
from utils import PolyLR

"""
Defines the model
"""

class SegFormer(pl.LightningModule):

    def __init__(self):

        # Configure the model
        super().__init__()

        model_config = SegformerConfig.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, semantic_loss_ignore_index = config.IGNORE_INDEX, return_dict=False)
        self.model = SegformerForSemanticSegmentation(model_config) # this does not load the imagenet weights yet.
        self.model = self.model.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, return_dict=False)  # this loads the weights

        # Initialize the optimizer
        self.optimizer = torch.optim.AdamW(params=[
            {'params': self.model.segformer.parameters(), 'lr': config.LEARNING_RATE},
            {'params': self.model.decode_head.parameters(), 'lr': 10 * config.LEARNING_RATE},
        ], lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

        # Initialize the metrics
        self.metrics = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)


    def configure_optimizers(self):
        iterations_per_epoch = math.ceil(config.NUMBER_TRAIN_IMAGES / (config.BATCH_SIZE * len(config.DEVICES)))
        total_iterations = iterations_per_epoch * self.trainer.max_epochs
        scheduler = PolyLR(self.optimizer, max_iterations=total_iterations, power=0.9)
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]


    def forward(self, x):
        return self.model(x)
    

    def training_step(self, batch, batch_index):
        images, labels = batch

        loss, logits = self.model(images, labels.squeeze(dim=1).long()) # uses CE loss by default
        
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    

    def validation_step(self, batch, batch_index):
        images, labels = batch

        loss, logits = self.model(images, labels.squeeze(dim=1).long()) # uses CE loss by default
        
        upsampled_logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)    # upsample logits to input image size (SegFormer outputs h/4 and w/4 by default, see paper)

        preds = torch.softmax(upsampled_logits, dim=1)

        iou = self.metrics(preds, labels.squeeze(dim=1))
        
        self.log('iou', iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)