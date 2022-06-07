import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np

import albumentations as A

# import windowing

class CTDataset(Dataset):

    def __init__(self, data_dir, binary_classification, inference=False, mode="test", added=True):
        self.inference = inference
        self.data = []
        self.mode = mode
        self.transform = A.Compose([ 
             A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
        ])
        for i,f in enumerate(os.listdir(data_dir)):
            if self.inference:
                self.data.append([os.path.join(data_dir, f)])
                continue
            if (not added) and ("image" in f):
                continue 

            if f[:2] == "IN":
                self.data.append((os.path.join(data_dir, f), 0))
            elif f[:2] == "IS":
                self.data.append((os.path.join(data_dir, f), 1))
            elif f[:2] == "KA":
                if binary_classification:
                    self.data.append((os.path.join(data_dir, f), 1))
                else:
                    self.data.append((os.path.join(data_dir, f), 2))

      

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        img_path = self.data[index][0]
        # img = windowing.output(dcm_path).astype(np.uint8)
        img = cv2.imread(img_path,0)
        img = cv2.resize(img, (512,512))

        
        if self.inference:
            img = torch.tensor(img)
            return img, img_path

        if self.mode == "train":
            img = self.transform(image=img)["image"]

        img = torch.tensor(img)
        label = self.data[index][1]
        sample = img, label
        return sample
