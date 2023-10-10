import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import os
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import torch

"""
Defines utility functions
"""

class PolyLR(_LRScheduler):
    """LR = Initial_LR * (1 - iter / max_iter)^0.9"""
    def __init__(self, optimizer, max_iterations, power=0.9):
        self.current_iteration = 0
        self.max_iterations = max_iterations
        self.power = power
        super().__init__(optimizer)

    def get_lr(self):
        self.current_iteration += 1
        return [base_lr * (1 - self.current_iteration / self.max_iterations) ** self.power for base_lr in self.base_lrs]

# Maps 40 classes to 13 classes
def map_40_to_13(mask):
    mapping = {0:11, 1:4, 2:5, 3:0, 4:3, 5:8, 6:9, 7:11, 8:12,	9:5, 10:7, 11:5, 12:12, 13:9, 14:5, 15:12,	
               16:5, 17:6, 18:6, 19:4, 20:6, 21:2, 22:1, 23:5, 24:10, 25:6, 26:6, 27:6, 28:6, 29:6,	30:6, 
               31:5, 32:6, 33:6, 34:6, 35:6, 36:6, 37:6, 38:5, 39:6, 255:255}
    
    mask = mask.squeeze().numpy().astype(int)

    for r, c in np.ndindex(mask.shape):
        mask[r, c] = mapping[mask[r, c]]

    return torch.tensor(mask, dtype=torch.long).unsqueeze(0)

# Creates a tuple of image and GT mask
def visualize_img_gt(image, gt_mask, filename='test.png'):
    # Image
    # Convert Image from Tensor to Image
    image = image.permute(1, 2, 0).numpy()

    plt.figure(figsize=(16, 5))

    # Place Subplots
    # Leave everything as it is!!!
    # If then only adjust the wspace value!!!
    plt.subplots_adjust(left=0.005,
                        bottom=0,
                        right=0.84,
                        top=1,
                        wspace=0.75,
                        hspace=0.0)
    
    plt.subplot(1, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)

    # Definde Labels and Colors as Dictionary
    labels_and_colors = {'Bett' : 'lightblue',
                         'Bücher' : 'brown',
                         'Decke' : 'lightyellow',
                         'Stuhl' : 'orange',
                         'Fußboden' : 'magenta',
                         'Möbel' : 'blue',
                         'Objekte' : 'green',
                         'Bild' : 'red',
                         'Sofa' : 'purple',
                         'Tisch' : 'goldenrod',
                         'Fernseher' : 'lightgreen',
                         'Wand' : 'gray',
                         'Fenster' : 'lightgray',
                         'Nicht annotiert' : 'white'}

    # Create Colormap from Dictionary
    cmap = mcolors.ListedColormap(list(labels_and_colors.values()))

    # Ground Truth Mask
    # Convert Mask from Tensor to Image
    gt_mask = gt_mask.squeeze().numpy()
    # Set Unlabeled Pixels to Value 14 (For the colormap)
    gt_mask[gt_mask == 255] = 14

    plt.subplot(1, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(gt_mask, cmap=cmap, vmin=0, vmax=14)

    # Legend
    legend_elements = [mpatches.Patch(facecolor=labels_and_colors[label],
                             edgecolor='black',
                             label=label) for label in labels_and_colors]
    plt.legend(handles=legend_elements,
               loc='center left',
               bbox_to_anchor=(1, 0.5))
    
    # Set Legend Title
    plt.gca().get_legend().set_title('Annotationen')

    # Set the legend font size
    plt.gca().get_legend().get_title().set_fontsize('xx-large')
    
    # Make legend bold
    plt.setp(plt.gca().get_legend().get_title(), fontweight='bold')

    # Set the font size of the labels and font type
    for label in plt.gca().get_legend().get_texts():
        label.set_fontsize('xx-large')
        label.set_fontfamily('serif')

    # Make the font of my legend look like latex
    plt.gca().get_legend().get_title().set_fontfamily('serif')

    # Save Figure
    directory = './figures/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, filename))
    plt.close()


# Creates a triple of image, GT mask and segmented mask and saves it to directory figures
def visualize_img_gt_pr(image, gt_mask, pr_mask, filename='test.png'):
    # Image
    # Convert Image from Tensor to Image
    image = image.permute(1, 2, 0).numpy()

    plt.figure(figsize=(16, 5))

    # Place Subplots
    # Leave everything as it is!!!
    # If then only adjust the wspace value!!!
    plt.subplots_adjust(left=0.005,
                        bottom=0,
                        right=0.84,
                        top=1,
                        wspace=0.01,
                        hspace=0.0)
    
    plt.subplot(1, 3, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)

    # Definde Labels and Colors as Dictionary
    labels_and_colors = {'Bett' : 'lightblue',
                         'Bücher' : 'brown',
                         'Decke' : 'lightyellow',
                         'Stuhl' : 'orange',
                         'Fußboden' : 'magenta',
                         'Möbel' : 'blue',
                         'Objekte' : 'green',
                         'Bild' : 'red',
                         'Sofa' : 'purple',
                         'Tisch' : 'goldenrod',
                         'Fernseher' : 'lightgreen',
                         'Wand' : 'gray',
                         'Fenster' : 'lightgray',
                         'Nicht annotiert' : 'white'}

    # Create Colormap from Dictionary
    cmap = mcolors.ListedColormap(list(labels_and_colors.values()))

    # Ground Truth Mask
    # Convert Mask from Tensor to Image
    gt_mask = gt_mask.squeeze().numpy()
    # Set Unlabeled Pixels to Value 14 (For the colormap)
    gt_mask[gt_mask == 255] = 14

    plt.subplot(1, 3, 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(gt_mask, cmap=cmap, vmin=0, vmax=14)

    # Prediction Truth Mask
    # Convert Mask from Tensor to Image
    pr_mask = pr_mask.squeeze().numpy()
    # Set Unlabeled Pixels to Value 14 (For the colormap)
    pr_mask[pr_mask == 255] = 14
    plt.subplot(1, 3, 3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(pr_mask, cmap=cmap, vmin=0, vmax=14)

    # Legend
    legend_elements = [mpatches.Patch(facecolor=labels_and_colors[label],
                             edgecolor='black',
                             label=label) for label in labels_and_colors]
    plt.legend(handles=legend_elements,
               loc='center left',
               bbox_to_anchor=(1, 0.5))
    
    # Set Legend Title
    plt.gca().get_legend().set_title('Annotationen')

    # Set the legend font size
    plt.gca().get_legend().get_title().set_fontsize('xx-large')
    
    # Make legend bold
    plt.setp(plt.gca().get_legend().get_title(), fontweight='bold')

    # Set the font size of the labels and font type
    for label in plt.gca().get_legend().get_texts():
        label.set_fontsize('xx-large')
        label.set_fontfamily('serif')

    # Make the font of my legend look like latex
    plt.gca().get_legend().get_title().set_fontfamily('serif')

    # Save Figure
    directory = './figures/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, filename))
    plt.close()