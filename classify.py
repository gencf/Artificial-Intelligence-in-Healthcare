import torch
from torch import nn 
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset
import cv2
import os
import shutil

class CTDataset(Dataset):

    def __init__(self, data_dir, binary_classification, inference=False):
        self.inference = inference
        self.data = []
        for i,f in enumerate(os.listdir(data_dir)):
            if self.inference:
                self.data.append([os.path.join(data_dir, f)])
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
        img = cv2.imread(img_path,0)
        img = cv2.resize(img, (512,512))
        img = torch.tensor(img)
        
        if self.inference:
            return img, img_path

        label = self.data[index][1]
        sample = img, label
        return sample


def classify(png_path, mask_path, iskemi_data_path, kanama_data_path, model_path):
    os.makedirs(os.path.join(iskemi_data_path,"PNG"), exist_ok=True)
    os.makedirs(os.path.join(iskemi_data_path,"MASKS"), exist_ok=True)
    os.makedirs(os.path.join(kanama_data_path,"PNG"), exist_ok=True)
    os.makedirs(os.path.join(kanama_data_path,"MASKS"), exist_ok=True)
    os.makedirs(os.path.join(iskemi_data_path, "RESULTS"), exist_ok=True)
    os.makedirs(os.path.join(kanama_data_path, "RESULTS"), exist_ok=True)


    out_features = 2
    batch_size = 1
    net = models.resnet18(pretrained=True)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.fc = nn.Linear(in_features=512, out_features=out_features, bias=True)

    net.load_state_dict(torch.load(model_path))

    net.cuda()
    net.eval()

    test_data = CTDataset(png_path, binary_classification=True, inference=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        for batch_idx,(X,file_name) in enumerate(test_loader):
            X = torch.reshape(X, [batch_size, 1, 512, 512]).cuda().float()
            pred = net(X)
            file_name = file_name[0].split("\\")[-1]
            label = pred.argmax(1).cpu().numpy()[0]
            
            if label == 0:  #iskemi
                shutil.copy(os.path.join(png_path, file_name), os.path.join(iskemi_data_path, "PNG", file_name))
                shutil.copy(os.path.join(mask_path, file_name), os.path.join(iskemi_data_path, "MASKS", file_name))
                
            elif label == 1:  #kanama
                shutil.copy(os.path.join(png_path, file_name), os.path.join(kanama_data_path, "PNG", file_name))
                shutil.copy(os.path.join(mask_path, file_name), os.path.join(kanama_data_path, "MASKS", file_name))                
    
