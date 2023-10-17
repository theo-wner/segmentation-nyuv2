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

class MiT(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # This line has to be added, ortherwise an error will be raised
        ssl._create_default_https_context = ssl._create_unverified_context

        # Initialize the model
        self.model = smp.Unet(
            encoder_name='mit_b5',
            encoder_weights='imagenet',
            in_channels=3,
            classes=config.NUM_CLASSES,
        )
        
        # Initialize the loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)

        # Initialize the optimizer
        self.optimizer = torch.optim.AdamW(params=[
            {'params': self.model.encoder.parameters(), 'lr': config.LEARNING_RATE},
            {'params': self.model.decoder.parameters(), 'lr': 10 * config.LEARNING_RATE},
            {'params': self.model.segmentation_head.parameters(), 'lr': 10 * config.LEARNING_RATE},
        ], lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

        # Initialize the metrics
        self.train_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        self.val_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        self.test_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        scores = self.model(images)
        loss = self.criterion(scores, masks.squeeze().long())
        train_iou = self.train_iou(torch.softmax(scores, dim=1), masks.squeeze())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_iou', train_iou, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        scores = self.model(images)
        val_iou = self.val_iou(torch.softmax(scores, dim=1), masks.squeeze())
        self.log('val_iou', val_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        images, masks = batch
        scores = self.model(images)
        test_iou = self.test_iou(torch.softmax(scores, dim=1), masks.squeeze())
        self.log('test_iou', test_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        iterations_per_epoch = math.ceil(config.NUMBER_TRAIN_IMAGES / (config.BATCH_SIZE * len(config.DEVICES)))
        total_iterations = iterations_per_epoch * self.trainer.max_epochs
        scheduler = PolyLR(self.optimizer, max_iterations=total_iterations, power=0.9)
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    

class SegFormer(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Ich habe für config.BACKBONE bisher 'b2' genutzt. Wenn das Training zu lange dauert oder die GPU Memory nicht ausreicht, könnte man aber auch 'b0' oder 'b1' nehmen.
        model_config = SegformerConfig.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, return_dict=False)
        self.model = SegformerForSemanticSegmentation(model_config) # this does not load the imagenet weights yet.
        self.model = self.model.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, return_dict=False)  # this loads the weights

        self.optimizer = torch.optim.AdamW(params=[
            {'params': self.model.segformer.parameters(), 'lr': config.LEARNING_RATE},
            {'params': self.model.decode_head.parameters(), 'lr': 10 * config.LEARNING_RATE},
        ], lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

        self.train_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        self.val_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)


    def training_step(self, batch, batch_index):
        images, labels = batch

        loss, logits = self.model(images, labels.squeeze(dim=1).long())    # uses CE loss by default
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    

    def validation_step(self, batch, batch_index):
        images, labels = batch

        loss, logits = self.model(images, labels.squeeze(dim=1).long())
        
        upsampled_logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)    # upsample logits to input image size (SegFormer outputs h/4 and w/4 by default, see paper)
        self.val_iou(torch.softmax(upsampled_logits, dim=1), labels.squeeze(dim=1))
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_iou', self.val_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    
    def configure_optimizers(self):
        iterations_per_epoch = math.ceil(config.NUMBER_TRAIN_IMAGES / (config.BATCH_SIZE * len(config.DEVICES)))
        total_iterations = iterations_per_epoch * self.trainer.max_epochs
        scheduler = PolyLR(self.optimizer, max_iterations=total_iterations, power=0.9)
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]