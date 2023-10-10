import torch
from model import DeepLab
from dataset import NYUv2Dataset
from utils import visualize_img_gt_pr

"""
Tests the model
"""

# Initialize the model
model = DeepLab()
model = model.load_from_checkpoint('lightning_logs/version_0/checkpoints/epoch=0-step=50.ckpt')

# Dataset
dataset = NYUv2Dataset(split='test')
image, mask = dataset[1]

# Push image to GPU
image = image.cuda()

# Prediction
model.eval()
with torch.no_grad():
    pred = model(image.unsqueeze(0))
    pred = torch.argmax(pred, dim=1)

# Push to CPU
image = image.cpu()
pred = pred.cpu()

# Visualization
visualize_img_gt_pr(image, mask, pred)

