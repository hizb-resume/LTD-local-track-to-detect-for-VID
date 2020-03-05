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
    weight_dir="siameseWeight2/"
    start_epoch=0
    train_batch_size = 256
    train_number_epochs = 100

class SiameseNetworkDataset(Dataset):

    def __init__(self, train_db, transform=None, should_invert=True):
        # self.imageFolderDataset = imageFolderDataset
        self.train_db = train_db
        self.transform = transform
        self.should_invert = should_invert
        self.lens=len(self.train_db['gt'])

    def __getitem__(self, index):

        # img0_tuple = random.choice(self.imageFolderDataset.imgs)
        img0_path=self.train_db['img_files'][index]
        img0_tackid=self.train_db['trackid'][index]
        img0_vidid = self.train_db['vid_id'][index]
        img0_gt = self.train_db['gt'][index]

        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        label=0
        if should_get_same_class:
            label=1
            letf_bd = index - 1000
            right_bd = index + 1000
            if letf_bd < 0:
                letf_bd = 0
            if right_bd > (self.lens - 2):
                right_bd=self.lens - 1
            ij=0
            while True:
                ij+=1
                # keep looping till the same class image is found
                idx = random.randint(letf_bd, right_bd)
                if idx==index:
                    continue
                img1_tackid = self.train_db['trackid'][idx]
                img1_vidid = self.train_db['vid_id'][idx]

                if img0_vidid==img1_vidid and img0_tackid==img1_tackid:
                    img1_path = self.train_db['img_files'][idx]
                    img1_gt = self.train_db['gt'][idx]
                    break
                elif ij<1000:
                    continue
                else:
                    img1_path = self.train_db['img_files'][idx]
                    img1_gt = self.train_db['gt'][idx]
                    label=0
                    break
        else:
            while True:
                # keep looping till a different class image is found
                # idx = random.randint(0, self.lens - 1)
                should_get_same_vid = random.randint(0, 1)
                if should_get_same_vid:
                    letf_bd = index - 500
                    right_bd = index + 500
                    if letf_bd < 0:
                        letf_bd = 0
                    if right_bd > (self.lens - 2):
                        right_bd = self.lens - 1
                    idx = random.randint(letf_bd, right_bd)
                else:
                    idx = random.randint(0, self.lens - 1)

                if idx == index:
                    continue
                img1_tackid = self.train_db['trackid'][idx]
                img1_vidid = self.train_db['vid_id'][idx]

                if img0_vidid == img1_vidid and img0_tackid == img1_tackid:
                    continue
                else:
                    img1_path = self.train_db['img_files'][idx]
                    img1_gt = self.train_db['gt'][idx]
                    break


        img0 = Image.open(img0_path)
        img1 = Image.open(img1_path)
        img0 = img0.crop(img0_gt)
        img1 = img1.crop(img1_gt)
        img0 = img0.convert("L")    #convert to grayscale img
        img1 = img1.convert("L")

        # if index%5==0:
        #     if label==0:
        #         s='false'
        #     else:
        #         s='True'
        #     img0.save("temimg/%d-%s-0.JPEG"%(index,s))
        #     img1.save("temimg/%d-%s-%d.JPEG" %(index,s,idx))

        if self.should_invert:  #Inverts binary images in black and white
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([label], dtype=np.float32))

    def __len__(self):
        return self.lens

if __name__ == "__main__" :
    # folder_dataset = dset.ImageFolder(root=Config.training_dir)
    # siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
    #                                         transform=transforms.Compose([transforms.Resize((100,100)),
    #                                                                       transforms.ToTensor()
    #                                                                       ])
    #                                        ,should_invert=False)
    pass
