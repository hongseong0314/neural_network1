import re
import os
import random
# import cv2
import numpy as np
import pandas as pd
import torch
import json
# from endecoder import *
import torchvision.transforms.functional as F

from PIL import Image
from torchvision import transforms

class DisDataset(torch.utils.data.Dataset):
    """
    dir_root : 데이터폴더 경로
    files : 가져올 데이터 csv
    mode : 불러 올 데이터(train or test)
    """
    def __init__(self,
                 root,
                 files,
                 mode='train',
                 img_size=224,
                 ):
        self.dir_root = root
        self.files = files
        self.mode = mode
        self.train_mode = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                            ])
        """
        transforms.RandomAffine((20)),
        transforms.RandomRotation(90),
        
        """
        self.test_mode = transforms.Compose([
                        transforms.Resize((img_size, img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                            ])

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        data = self.files.iloc[index, ]
        
        crop_label = torch.zeros((11))
        crop_label[data['crop']] += 1
        disease_label = torch.zeros((30))
        disease_label[data['disease']] += 1

        image_path = os.path.join(self.dir_root, data['path'])
        image = Image.open(image_path).convert('RGB')
        sample = {'image': image, 'crop_label': crop_label, 'disease_label':torch.argmax(disease_label)}

        # train mode transform
        if self.mode == 'train':
            sample['image'] = self.train_mode(sample['image'])

        # test mode transform
        elif self.mode == 'test' or self.mode == 'valid':
            sample['image'] = self.test_mode(sample['image'])
        return sample
