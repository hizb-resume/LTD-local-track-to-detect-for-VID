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
import os
from datasets.SiameseDataset import Config,SiameseNetworkDataset
from models.SiameseNet import SiameseNetwork,ContrastiveLoss

if __name__ == "__main__" :
    if not os.path.exists(Config.weight_dir):
        os.makedirs(Config.weight_dir)
    folder_dataset = dset.ImageFolder(root=Config.training_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                          transforms.ToTensor()
                                                                          ])
                                            , should_invert=False)
    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=8,
                                  batch_size=Config.train_batch_size)
    net = SiameseNetwork().cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    # resume = 'siameseWeight/SiameseNet_epoch1.pth'
    resume=False
    if resume:
        net.load_weights(resume)
        checkpoint = torch.load(resume)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(Config.start_epoch, Config.train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
        print('Saving state, epoch:', epoch)
        torch.save({
            'epoch': epoch ,
            'net_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(Config.weight_dir) +
           'SiameseNet_epoch' + repr(epoch) + '_final.pth')
    # show_plot(counter, loss_history)