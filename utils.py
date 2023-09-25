import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import os
import numpy as np


# Creates a triple of image, GT mask and segmented mask and saves it to directory figures
def visualize_img_mask(image, gt_mask, pr_mask, filename='test.png'):
    # Image
    # Convert Image from Tensor to Image
    image = image.permute(1, 2, 0).numpy()

    plt.figure(figsize=(16, 5))

    # Place Subplots in the middle
    plt.subplots_adjust(left=0.02,
                        bottom=0.02,
                        right=0.90,
                        top=0.98,
                        wspace=0.05,
                        hspace=0.05)
    
    plt.subplot(1, 3, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title('Image')
    plt.imshow(image)

    # Definde Labels and Colors as Dictionary
    labels_and_colors = {'bed' : 'lightblue',
                         'books' : 'brown',
                         'ceiling' : 'lightyellow',
                         'chair' : 'orange',
                         'floor' : 'magenta',
                         'furniture' : 'blue',
                         'objects' : 'green',
                         'picture' : 'red',
                         'sofa' : 'purple',
                         'table' : 'goldenrod',
                         'tv' : 'lightgreen',
                         'wall' : 'gray',
                         'window' : 'lightgray',
                         'unlabeled' : 'white'}

    # Create Colormap from Dictionary
    cmap = mcolors.ListedColormap(list(labels_and_colors.values()))

    # Ground Truth Mask
    # Convert Mask from Tensor to Image
    gt_mask = gt_mask.squeeze().numpy()
    # Restore original range from 0 to 255 (Tensors are from 0 to 1)
    gt_mask = gt_mask * 255
    # Convert to Integer
    gt_mask = gt_mask.astype(int)
    # Set Unlabeled Pixels to Value 14 (For the colormap)
    gt_mask[gt_mask == 255] = 14

    plt.subplot(1, 3, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('Ground Truth')
    plt.imshow(gt_mask, cmap=cmap, vmin=0, vmax=14)

    # Predicted Mask
    # Convert Mask from Tensor to Image
    pr_mask = pr_mask.squeeze().numpy()
    # Restore original range from 0 to 255 (Tensors are from 0 to 1)
    pr_mask = pr_mask * 255
    # Convert to Integer
    pr_mask = pr_mask.astype(int)
    # Set Unlabeled Pixels to Value 14
    pr_mask[pr_mask == 255] = 14
    
    plt.subplot(1, 3, 3)
    plt.xticks([])
    plt.yticks([])
    plt.title('Prediction')
    plt.imshow(pr_mask, cmap=cmap, vmin=0, vmax=14)

    # Legend
    legend_elements = [mpatches.Patch(facecolor=labels_and_colors[label],
                             edgecolor='black',
                             label=label) for label in labels_and_colors]
    plt.legend(handles=legend_elements,
               loc='center left',
               bbox_to_anchor=(1, 0.5))

    # Save Figure
    directory = './figures/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, filename))
    plt.close()