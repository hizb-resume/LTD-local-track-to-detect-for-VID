# matlab code:
# https://github.com/hellbell/ADNet/blob/3a7955587b5d395401ebc94a5ab067759340680d/utils/get_extract_regions.m
# other reference: https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

import numpy as np
import types
import torch
from torchvision import transforms
import cv2
import types
from numpy import random
import torch.nn.functional as F
from PIL import Image

class ToTensor(object):
    def __call__(self, cvimage, box=None, action_label=None, conf_label=None):
        return torch.from_numpy(cvimage.astype(np.float32)).unsqueeze(0).permute(0, 3,1,2), box, action_label, conf_label

class ToTensor2(object):
    def __call__(self, cvimage, box=None, action_label=None, conf_label=None):
        return cvimage.permute(2, 0, 1), box, action_label, conf_label

class ToTensor3(object):
    def __call__(self, cvimage, box=None):
        return torch.from_numpy(cvimage.astype(np.float32)).unsqueeze(0), box

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, box=None, action_label=None, conf_label=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), box, action_label, conf_label

class SubtractMeans2(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, box=None, action_label=None, conf_label=None):
        # image = image.astype(np.float32)
        im =image- self.mean
        return im, box, action_label, conf_label


class CropRegion(object):
    def __call__(self, image, box, action_label=None, conf_label=None):
        image = np.array(image)
        box = np.array(box)
        if box is not None:
            center = box[0:2] + 0.5 * box[2:4]
            wh = box[2:4] * 1.4  # multiplication = 1.4
            box_lefttop = center - 0.5 * wh
            box_rightbottom = center + 0.5 * wh
            box_ = [
                max(0, box_lefttop[0]),
                max(0, box_lefttop[1]),
                min(box_rightbottom[0], image.shape[1]),
                min(box_rightbottom[1], image.shape[0])
            ]

            im = image[int(box_[1]):int(box_[3]), int(box_[0]):int(box_[2]), :]
        else:
            im = image[:, :, :]

        return im.astype(np.float32), box, action_label, conf_label

class CropRegion2(object):
    def __call__(self, image, box, action_label=None, conf_label=None):
        # image = np.array(image)
        box = np.array(box)
        if box is not None:
            center = box[0:2] + 0.5 * box[2:4]
            wh = box[2:4] * 1.4  # multiplication = 1.4
            box_lefttop = center - 0.5 * wh
            box_rightbottom = center + 0.5 * wh
            box_ = [
                max(0, box_lefttop[0]),
                max(0, box_lefttop[1]),
                min(box_rightbottom[0], image.shape[1]),
                min(box_rightbottom[1], image.shape[0])
            ]

            im = image[int(box_[1]):int(box_[3]), int(box_[0]):int(box_[2]), :]
        else:
            im = image[:, :, :]

        return im, box, action_label, conf_label

class CropRegion3(object):
    def __call__(self, image,img2, box):
        # image = np.array(image)
        box = np.array(box)
        if box is not None:
            center = box[0:2] + 0.5 * box[2:4]
            wh = box[2:4] * 1  # multiplication = 1.4
            box_lefttop = center - 0.5 * wh
            box_rightbottom = center + 0.5 * wh
            box_ = [
                max(0, box_lefttop[0]),
                max(0, box_lefttop[1]),
                min(box_rightbottom[0], image.shape[1]),
                min(box_rightbottom[1], image.shape[0])
            ]

            im = image[int(box_[1]):int(box_[3]), int(box_[0]):int(box_[2]), :]
        else:
            im = image[:, :, :]

        return im,im, box

# crop "multiplication" times of the box width and height
class CropRegion_withContext(object):
    def __init__(self, multiplication=None):
        if multiplication is None:
            multiplication = 1.4  # same with default CropRegion
        assert multiplication >= 1, "multiplication should more than 1 so the object itself is not cropped"
        self.multiplication = multiplication

    def __call__(self, image, box, action_label=None, conf_label=None):
        image = np.array(image)
        box = np.array(box)
        if box is not None:
            center = box[0:2] + 0.5 * box[2:4]
            wh = box[2:4] * self.multiplication
            box_lefttop = center - 0.5 * wh
            box_rightbottom = center + 0.5 * wh
            box_ = [
                max(0, box_lefttop[0]),
                max(0, box_lefttop[1]),
                min(box_rightbottom[0], image.shape[1]),
                min(box_rightbottom[1], image.shape[0])
            ]

            im = image[int(box_[1]):int(box_[3]), int(box_[0]):int(box_[2]), :]
        else:
            im = image[:, :, :]

        return im.astype(np.float32), box, action_label, conf_label


class ResizeImage(object):
    def __init__(self, inputSize):
        self.inputSize = inputSize  # network's input size (which is the output size of this function)

    def __call__(self, image, box, action_label=None, conf_label=None):
        im = cv2.resize(image, dsize=tuple(self.inputSize[:2]))
        return im.astype(np.float32), box, action_label, conf_label

class ResizeImage2(object):
    def __init__(self, inputSize):
        self.inputSize = inputSize  # network's input size (which is the output size of this function)

    def __call__(self, image, box, action_label=None, conf_label=None):
        # im = cv2.resize(image, dsize=tuple(self.inputSize[:2]))
        im=image.unsqueeze(0)
        im = F.interpolate(im,tuple(self.inputSize[:2]))
        # im = im.squeeze(0)
        return im.permute(0,1, 3,2), box, action_label, conf_label

class ResizeImage3(object):
    def __call__(self, image,img2, box):
        # im = cv2.cvtColor(cv2.resize(image, dsize=(100,100),interpolation=cv2.INTER_CUBIC),cv2.COLOR_BGR2GRAY)
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # im=image.unsqueeze(0)
        # im = F.interpolate(im,tuple(self.inputSize[:2]))
        # im = im.squeeze(0)
        return im,image, box

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        # >>> augmentations.Compose([
        # >>>     transforms.CenterCrop(10),
        # >>>     transforms.ToTensor(),
        # >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, box=None, action_label=None, conf_label=None):
        for t in self.transforms:
            img, box, action_label, conf_label = t(img, box, action_label, conf_label)
        return img, box, action_label, conf_label

class Compose3(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        # >>> augmentations.Compose([
        # >>>     transforms.CenterCrop(10),
        # >>>     transforms.ToTensor(),
        # >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1,img2, box=None):
        for t in self.transforms:
            img1, img2,box = t(img1,img2, box)
        return img1,img2, box


class ADNet_Augmentation(object):
    def __init__(self, opts):
        self.augment = Compose([
            SubtractMeans(opts['means']),
            CropRegion(),
            ResizeImage(opts['inputSize']),
            ToTensor()
        ])

    def __call__(self, img, box, action_label=None, conf_label=None):
        return self.augment(img, box, action_label, conf_label)


class ADNet_Augmentation2(object):
    def __init__(self, opts,mean):
        self.augment = Compose([
            CropRegion2(),
            SubtractMeans2(mean),
            ToTensor2(),
            ResizeImage2(opts['inputSize'])
        ])

    def __call__(self, img, box, action_label=None, conf_label=None):
        return self.augment(img, box, action_label, conf_label)

class ADNet_Augmentation3(object):
    def __init__(self,transform3_adition):
        self.augment = Compose3([
            CropRegion3(),
            ResizeImage3(),
            # ToTensor3()
        ])
        self.transform3_adition=transform3_adition

    def __call__(self, img, box):
        # return self.augment(img, box)
        img,img_crop_origin,_=self.augment(img,img, box)
        img=Image.fromarray(img)
        return self.transform3_adition(img).unsqueeze(0),img_crop_origin,box

class ADNet_Augmentation4(object):
    def __init__(self,transform3_adition):
        self.augment = Compose3([
            CropRegion3(),
            ResizeImage3(),
            # ToTensor3()
        ])
        self.transform3_adition=transform3_adition

    def __call__(self, img, box):
        # return self.augment(img, box)
        img,img_crop_origin,_=self.augment(img,img, box)
        img=Image.fromarray(img)
        return self.transform3_adition(img),img_crop_origin,box