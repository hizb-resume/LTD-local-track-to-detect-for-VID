import torch
import torchvision
from utils.my_util import aHash,Hamming_distance
# print(torch.cuda.is_available())
#
# a = torch.Tensor(5,3)
# a=a.cuda()
# print(a)
# layers=[1,2,3,4,5,6,7,8,9]

# layers="hello"
# print(layers[-2:])
# for l in layers[::-1]:
#     print(l)

from PIL import Image #use PIL to processs img
import os
import numpy as np
#import cv2     #import when use opencv to process img



if __name__ == "__main__" :
    #PIL
    image1 = Image.open('image1.png')
    image2 = Image.open('image2.png')
    #reduce size and grayscale
    image1=np.array(image1.resize((8, 8),Image.ANTIALIAS).convert('L'),'f')
    image2=np.array(image2.resize((8, 8),Image.ANTIALIAS).convert('L'),'f')

    #opencv
    #img1 = cv2.imread('image1')
    #img2 = cv2.imread('image2')
    #reduce size and grayscale
    #image1=cv2.cvtColor(cv2.resize(img1,(8, 8), interpolation=cv2.INTER_CUBIC)
    #image2=cv2.cvtColor(cv2.resize(img2,(8, 8), interpolation=Cv2.INTER_CUBIC)
    hash1 = aHash(image1)
    hash2 = aHash(image2)
    dist = Hamming_distance(hash1, hash2)
    #convert distance to similarity
    similarity = 1-dist * 1.0 / 64
    print('dist is %d' % dist)
    print('similarity is %d' % similarity)