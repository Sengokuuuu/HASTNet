import random
import math
import albumentations as A
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import os

join = os.path.join


def make_odd(num):
    num = math.ceil(num)
    if num % 2 == 0:
        num += 1
    return num


def apply_transform(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])
    transformed = transform(image=image, mask=mask)
    return transformed['image'], transformed['mask']


class SDDataset(Dataset):
    def __init__(self, data_root, train=True, resize_size=[384, 384], label_resize_size=[], train_ratio=1.0):
        self.train = train
        self.data_root = data_root
        self.resize_size = resize_size
        if len(label_resize_size) <= 0:
            self.label_resize_size = resize_size
        else:
            self.label_resize_size = label_resize_size
        self.img_files = []
        self.gt_files = []
        self.image_size = resize_size[0]
        
        if train:
            self.scan_list = os.listdir(os.path.join(self.data_root, "train", "images"))
            self.scan_list.sort()
            self.scan_list = self.scan_list[:int(len(self.scan_list) * train_ratio)]
            for scan in self.scan_list:
                self.img_files.append(os.path.join(self.data_root, "train", "images", scan))
                gt_file = os.path.join(self.data_root, "train", "targets", scan.replace(".jpg", ".png"))
                self.gt_files.append(gt_file)
        else:
            self.scan_list = os.listdir(os.path.join(self.data_root, "val", "images"))
            self.scan_list.sort()
            for scan in self.scan_list:
                self.img_files.append(os.path.join(self.data_root, "val", "images", scan))
                gt_file = os.path.join(self.data_root, "val", "targets", scan.replace(".jpg", ".png"))
                self.gt_files.append(gt_file)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img = cv2.imread(self.img_files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.resize_size)
        
        gt2D = cv2.imread(self.gt_files[index], cv2.IMREAD_GRAYSCALE)
        gt2D = cv2.resize(gt2D, self.label_resize_size, cv2.INTER_NEAREST)
        
        if self.train:
            img, gt2D = apply_transform(img, gt2D)
            
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        gt2D = gt2D.astype(np.uint8)
        
        return torch.tensor(img).float(), torch.tensor(gt2D[None, :, :]).long(), self.img_files[index], self.gt_files[index]


if __name__ == "__main__":
    dataset = SDDataset(data_root='dataset/', train=True)
    print(len(dataset))
    for i in range(5):
        img, gt, img_path, gt_path = dataset[i]
        print("img shape:", img.shape)
        print("gt shape:", gt.shape)
        print("img path:", img_path)
        print("gt path:", gt_path)