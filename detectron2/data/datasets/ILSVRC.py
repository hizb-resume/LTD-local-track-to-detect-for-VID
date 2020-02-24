# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

import os
import numpy as np
import cv2
import random
import json
import xml.etree.ElementTree as ET
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from google.colab.patches import cv2_imshow

__all__ = ["get_ILSVRC_dicts","register_ILSVRC"]


# CLASS_NAMES = [
#     {'n02691156': 1 },{'n02419796': 2 },{'n02131653': 3 },
#     {'n02834778': 4 },{'n01503061': 5 },{'n02924116': 6 },
#     {'n02958343': 7 },{'n02402425': 8 },{'n02084071': 9 },
#     {'n02121808': 10 },{'n02503517': 11 },{'n02118333': 12 },
#     {'n02510455': 13 },{'n02342885': 14 },{'n02374451': 15 },
#     {'n02129165': 16 },{'n01674464': 17 },{'n02484322': 18 },
#     {'n03790512': 19 },{'n02324045': 20 },{'n02509815': 21 },
#     {'n02411705': 22 },{'n01726692': 23 },{'n02355227': 24 },
#     {'n02129604': 25 },{'n04468005': 26 },{'n01662784': 27 },
#     {'n04530566': 28 },{'n02062744': 29 },{'n02391049': 30 }
# ]
# CLASS_NAMES = [
#     'n02691156', 'n02419796', 'n02131653', 'n02834778', 'n01503061',
#     'n02924116', 'n02958343', 'n02402425', 'n02084071', 'n02121808',
#     'n02503517', 'n02118333', 'n02510455', 'n02342885', 'n02374451',
#     'n02129165', 'n01674464', 'n02484322', 'n03790512', 'n02324045',
#     'n02509815', 'n02411705', 'n01726692', 'n02355227', 'n02129604',
#     'n04468005', 'n01662784', 'n04530566', 'n02062744', 'n02391049'
# ]
CLASS_NAMES = [
    'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
    'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda',
    'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle',
    'rabbit', 'red_panda', 'sheep', 'snake', 'squirrel', 'tiger',
    'train', 'turtle', 'watercraft', 'whale', 'zebra'
]

#get_dicts example:
def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def code_to_code_chall(argument):
    switcher = {
        'n02691156': 1,
        'n02419796': 2,
        'n02131653': 3,
        'n02834778': 4,
        'n01503061': 5,
        'n02924116': 6,
        'n02958343': 7,
        'n02402425': 8,
        'n02084071': 9,
        'n02121808': 10,
        'n02503517': 11,
        'n02118333': 12,
        'n02510455': 13,
        'n02342885': 14,
        'n02374451': 15,
        'n02129165': 16,
        'n01674464': 17,
        'n02484322': 18,
        'n03790512': 19,
        'n02324045': 20,
        'n02509815': 21,
        'n02411705': 22,
        'n01726692': 23,
        'n02355227': 24,
        'n02129604': 25,
        'n04468005': 26,
        'n01662784': 27,
        'n04530566': 28,
        'n02062744': 29,
        'n02391049': 30}
    return switcher.get(argument, "nothing")

def get_xml_img_info(xmlpath):
    img_info = {
        #'trackid':[],
        'name':[],
        'imgsize': [],
        'gts': []
    }
    in_file = open(xmlpath)
    tree = ET.parse(in_file)
    root = tree.getroot()
    imgsize = [0, 0]
    siz = root.find('size')
    imgsize[1] = int(siz.find('width').text)
    imgsize[0] = int(siz.find('height').text)
    gts = []
    for obj in root.iter('object'):
        # tckid=obj.find('trackid').text
        nm=obj.find('name')
        # nm=vid_classes.code_to_class_string(str(nm.text))
        bb = obj.find('bndbox')
        gt = [0, 0, 0, 0]
        gt[0] = int(bb.find('xmin').text)
        gt[1] = int(bb.find('ymin').text)
        gt[2] = int(bb.find('xmax').text)
        gt[3] = int(bb.find('ymax').text)
        # gt[2] = gt[2] - gt[0]
        # gt[3] = gt[3] - gt[1]

        # img_info['trackid'].append(tckid)
        img_info['name'].append(nm)
        gts.append(gt)

        # track_id=int(obj.find('trackid').text)
        # if track_id==0:
        #     break
    in_file.close()
    img_info['imgsize'] =imgsize
    img_info['gts']=gts
    return img_info

def get_ILSVRC_dicts(path_root,img_dir,det_or_vid,train_or_val):
    json_file = os.path.join(path_root, img_dir)
    # with open(json_file) as f:
    #     path_info = json.load(f)
    pathf = open(json_file, "r")
    path_info = pathf.readlines()
    pathf.close()

    path_info = [line.split(' ')[0] for line in path_info]
    dataset_dicts = []
    for idx, v in enumerate(path_info):
        record = {}

        filename = os.path.join(path_root,"Data",det_or_vid,train_or_val, v,".JPEG")
        gtfilepath= os.path.join(path_root,"Annotations",det_or_vid,train_or_val, v,".xml")
        img_info=get_xml_img_info(gtfilepath)

        height = img_info['imgsize'][0]
        width = img_info['imgsize'][1]

        # height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        # annos = v["regions"]
        objs = []
        # for _, anno in img_info[]:
        for i in range(len(img_info['name'])):
            # assert not anno["region_attributes"]
            # anno = anno["shape_attributes"]
            # px = anno["all_points_x"]
            # py = anno["all_points_y"]
            # poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            # poly = [p for x in poly for p in x]

            obj = {
                "bbox": img_info['gts'][i],
                "bbox_mode": BoxMode.XYXY_REL,
                # "segmentation": [poly],
                "category_id": code_to_code_chall(str(img_info['name'][i])),
                # "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def register_ILSVRC():
    for c in ["DET","VID"]:
        for d in ["train", "val"]:
            DatasetCatalog.register("ILSVRC_" +c+"_"+ d, lambda d=d: get_ILSVRC_dicts(
                "../datasets/data/ILSVRC/",("ImageSets/VID/" + d+".txt"),c,d))
            MetadataCatalog.get("ILSVRC_" +c+"_"+ d).set(thing_classes=CLASS_NAMES)

if __name__ == "__main__":
    register_ILSVRC()
    for c in ["DET","VID"]:
        for d in ["train", "val"]:
            dataset_dicts = get_ILSVRC_dicts(
                "../datasets/data/ILSVRC/", ("ImageSets/VID/" + d + ".txt"), c, d)
            ILSVRC_metadata = MetadataCatalog.get("ILSVRC_" +c+"_"+ d)
            for d in random.sample(dataset_dicts, 3):
                img = cv2.imread(d["file_name"])
                visualizer = Visualizer(img[:, :, ::-1], metadata=ILSVRC_metadata, scale=0.5)
                vis = visualizer.draw_dataset_dict(d)
                cv2_imshow(vis.get_image()[:, :, ::-1])