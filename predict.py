import torch
from model import DeepLab
from dataset import NYUv2Dataset

from utils import visualize_img_gt_pr, map_40_to_13
import config

"""
Predicts with the model
"""

if __name__ == '__main__':
    # Initialize the model (and load it to the CPU)
    model = DeepLab()

    if config.CPU_USAGE:
        checkpoint = torch.load("epoch=1999-step=200000.ckpt", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = model.load_from_checkpoint("epoch=1999-step=200000.ckpt")
    model.eval()

    # Dataset
    dataset = NYUv2Dataset(split='test')

    # Predict
    for i in range(50):
        image, mask = dataset[i]

        with torch.no_grad():
            pred = model(image.unsqueeze(0))
            pred = torch.argmax(pred, dim=1).squeeze()

        # Mask out the predictions where the original image is black in case you predict augmented train images
        pred[image.sum(dim=0) == 0] = 255

        # Visualization
        mask = map_40_to_13(mask)
        pred = map_40_to_13(pred)
        visualize_img_gt_pr(image, mask, pred, filename=f'test_{i}.png')

