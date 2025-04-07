import torch
import cv2
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import os
import random
import math


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, transform=None,valid=False):
        self.transform = transform
        self.Tensor = transforms.ToTensor()
        self.valid=valid
        if valid:
            self.root='E:/tumor_segmentation2/paper_write/Tumor_dataset2/images/val/'
            self.names=os.listdir(self.root)
        else:
            self.root='E:/tumor_segmentation2/paper_write/Tumor_dataset2/images/train/'
            self.names=os.listdir(self.root)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        '''

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        W_=640
        H_=360
        image_name=os.path.join(self.root,self.names[idx])
        
        image = cv2.imread(image_name)
        # label1 = cv2.imread(image_name.replace("images","segments").replace("jpg","png"), 0)
        # label2 = cv2.imread(image_name.replace("images","lane").replace("jpg","png"), 0)
        label1 = cv2.imread(image_name.replace("images","Whole"), 0)
        label2 = cv2.imread(image_name.replace("images","Core"), 0)   
        label1 = cv2.resize(label1, (W_, H_))
        label2 = cv2.resize(label2, (W_, H_))
        image = cv2.resize(image, (W_, H_))

        _,seg_b1 = cv2.threshold(label1,1,255,cv2.THRESH_BINARY_INV)
        _,seg_b2 = cv2.threshold(label2,1,255,cv2.THRESH_BINARY_INV)
        _,seg1 = cv2.threshold(label1,1,255,cv2.THRESH_BINARY)
        _,seg2 = cv2.threshold(label2,1,255,cv2.THRESH_BINARY)

        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        seg_b1 = self.Tensor(seg_b1)
        seg_b2 = self.Tensor(seg_b2)
        seg_da = torch.stack((seg_b1[0], seg1[0]),0)
        seg_ll = torch.stack((seg_b2[0], seg2[0]),0)
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)      
        return image_name,torch.from_numpy(image),(seg_da,seg_ll)