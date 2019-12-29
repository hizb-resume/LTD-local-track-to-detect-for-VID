import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

cfg = get_cfg()
cfg.merge_from_file("../configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
cfg.MODEL.WEIGHTS = "faster_rcnn_R_101_FPN_3x.pkl"
predictor = DefaultPredictor(cfg)

vidpath="./examples/jiaotong.mp4"
cap = cv2.VideoCapture(vidpath)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
i=0
length=length//3
d=length//5
while i<length:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(i))
    success, im = cap.read()
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2_imshow(v.get_image()[:, :, ::-1])
    cv2.imwrite('./examples/output/%d.jpg'%i,v.get_image(),[int(cv2.IMWRITE_JPEG_QUALITY),70])
    #cv2_imshow(im)
    i+=d




'''
im = cv2.imread("demo/input.jpg")
#cv2_imshow(im)
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
cfg.MODEL.WEIGHTS = "demo/mask_rcnn_R_50_FPN_3x.pkl"
predictor = DefaultPredictor(cfg)
outputs = predictor(im)
#outputs["instances"].pred_classes
#outputs["instances"].pred_boxes

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2_imshow(v.get_image()[:, :, ::-1])

'''

