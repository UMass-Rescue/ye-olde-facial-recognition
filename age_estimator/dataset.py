import torch
from torch.utils.data import Dataset

from PIL import Image

from pathlib import Path

import pandas as pd
import numpy as np


class CroppedAppaRealDataset(Dataset):
    """Custom Dataset for loading cropped APPA-REAL face images."""


    def __init__(self, csv_file, img_dir, transform=None):
        self.label_data = pd.read_csv(csv_file)
        
        # Modify file names to load cropped images instead of full images
        self.label_data['file_name'] = self.label_data['file_name'].astype(str) + '_face.jpg'

        self.img_dir = Path(img_dir)

        self.transform = transform


    def __getitem__(self, idx):
        img_path = self.img_dir.joinpath(self.label_data.iloc[idx, 0])
        
        img = Image.open(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        apparent_age = self.label_data.iloc[idx, 2]
        std_dev = self.label_data.iloc[idx, 3]
        final_age = apparent_age + np.random.randn() * std_dev

        return img, np.clip(round(final_age), 0, 100)


    def __len__(self):
        return len(self.label_data)
