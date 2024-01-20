# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class DRDataset(Dataset):
    def __init__(self, labels, root_dir, transform=None):
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = '{}.jpg'.format(self.labels.iloc[idx, 0])
        fullname = os.path.join(self.root_dir, img_name)
        image = Image.open(fullname)
        labels = self.labels.iloc[idx, 1].astype(np.int64)
        if self.transform:
            image = self.transform(image)
        return [image, labels]
