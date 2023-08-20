import torch
from collections import defaultdict
import os
from PIL import Image # For PNGs
import config
from skimage import io, exposure
import numpy as np

class CXR_Dataset(torch.utils.data.Dataset):
    """
        Class for loading the images and their corresponding labels.
        Parameters:
        df (pandas.DataFrame): DataFrame with image information.
        transform (callable): Data augmentation method.
        mode (str): Mode of the data ('train', 'test', 'val').
    """
    def __init__(
        self,
        dataframe,
        transforms=None,
    ):
        super().__init__()
        self.df = dataframe
        self.data = defaultdict(dict)
        self.transforms = transforms
        
        counter = 0
        for each in range(len(self.df)):
            self.data[counter] = {
                "image_path": os.path.join('/cxr', self.df.ImagePath.iloc[each]),
                "target1": self.df.Cardiomegaly.iloc[each],
                "target2": self.df.Pneumothorax.iloc[each],
                "target3": self.df["Pleural Effusion"].iloc[each],
            }
            counter += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # try:
        img_path = self.data[item]["image_path"]
        target1 = self.data[item]["target1"]
        target2 = self.data[item]["target2"]
        target3 = self.data[item]["target3"]
        
        target1 = torch.tensor(target1)
        target2 = torch.tensor(target2)
        target3 = torch.tensor(target3)
        
        #x = io.imread(img_path)
        x = Image.open(img_path).resize((256, 256), resample=Image.BILINEAR)
        x = np.array(x)
        x = exposure.equalize_hist(x) # Histogram equalization
        x = (((x - x.min()) / x.max() - x.min())*255).astype(np.uint8) # Converting 16 bit to 8 bit image # DONOT DO THESE # Normalize using some library
        x = np.stack((x, )*3) # Stack the Gray scaled 1 channel image 3 times to convert to 3 channel image
        x = np.transpose(x, (1, 2, 0)) # Convert Channel first to channel last
        
        x = self.transforms(x)
        return {"img": x, "target1": target1, "target2": target2, "target3": target3}

        # except Exception as e:
        #     print(f"Error loading image {img_path}: {e}")
        #     return None