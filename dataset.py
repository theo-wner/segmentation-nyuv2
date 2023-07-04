import os
import cv2
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import config

class NYUv2Dataset(Dataset):
    # split is 'train' or 'val', both splits are retrieved from the train set with different transforms
    def __init__(self, split='train'):
        self.root_dir = './data'
        self.image_set = split
        self.ignore_index = config.IGNORE_INDEX

        # Define the path of images and labels
        self.images_dir = os.path.join(self.root_dir, 'image', 'train')
        self.masks_dir = os.path.join(self.root_dir, 'seg13mod', 'train')

        # Read the list of image filenames
        self.filenames = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, index):
        # Load image and mask
        image = self._load_image(index)
        mask = self._load_mask(index)

        # In case of training, apply data augmentation
        if self.image_set == 'train':
            # Data augmentation
            pass

        # In case of validation, only resize
        elif self.image_set == 'val':
            # Resize
            pass

        return image, mask

    def _load_image(self, index):
        image_filename = os.path.join(self.images_dir, self.filenames[index])
        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = TF.to_tensor(image)
        return image
        
    def _load_mask(self, index):
        mask_filename = os.path.join(self.masks_dir, self.filenames[index])
        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
        mask = TF.to_tensor(mask)
        return mask
    
# Main function for testing
if __name__ == '__main__':
    dataset = NYUv2Dataset()

    image, mask = dataset[0]
    print(image.shape)
    print(mask.shape)



