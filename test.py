import torch
from model import DeepLab
from dataset import NYUv2Dataset
from utils import visualize_img_gt_pr, map_40_to_13

"""
Tests the model
"""

# Initialize the model
model = DeepLab()
model = model.load_from_checkpoint('tb_logs/nyuv2_40_classes/version_1/checkpoints/epoch=376-step=37700.ckpt')

# Dataset
dataset = NYUv2Dataset(split='test')

for i in range(50):
    image, mask = dataset[i]

    # Push image to GPU
    image = image.cuda(1)

    # Prediction
    model.eval()
    with torch.no_grad():
        pred = model(image.unsqueeze(0))
        pred = torch.argmax(pred, dim=1)

    # Push to CPU
    image = image.cpu()
    pred = pred.cpu()

    # Visualization
    mask = map_40_to_13(mask)
    pred = map_40_to_13(pred)
    visualize_img_gt_pr(image, mask, pred, filename=f'test_{i}.png')

