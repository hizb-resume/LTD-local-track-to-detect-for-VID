import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class Config():
    training_dir = "./data/faces/training/"
    testing_dir = "./data/faces/testing/"
    weight_dir="siameseWeight/"
    start_epoch=0
    train_batch_size = 64
    train_number_epochs = 100

class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, train_db, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.train_db = train_db
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

if __name__ == "__main__" :
    # folder_dataset = dset.ImageFolder(root=Config.training_dir)
    # siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
    #                                         transform=transforms.Compose([transforms.Resize((100,100)),
    #                                                                       transforms.ToTensor()
    #                                                                       ])
    #                                        ,should_invert=False)
    pass
