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
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.engine import DefaultTrainer, default_argument_parser, launch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
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

OBJ_COLORS=[
    [220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228],
    [0, 60, 100], [0, 80, 100], [0, 0, 70], [0, 0, 192], [250, 170, 30],
    [100, 170, 30], [220, 220, 0], [175, 116, 175], [250, 0, 30], [165, 42, 42],
    [255, 77, 255], [0, 226, 252], [182, 182, 255], [0, 82, 0], [120, 166, 157],
    [110, 76, 0], [174, 57, 255], [199, 100, 0], [72, 0, 118], [255, 179, 240],
    [0, 125, 92], [209, 0, 151], [188, 208, 182], [0, 220, 176], [255, 99, 164],
    [92, 0, 73], [133, 129, 255], [78, 180, 255], [0, 228, 0], [174, 255, 243],
    [45, 89, 255], [134, 134, 103], [145, 148, 174], [255, 208, 186],
    [197, 226, 255], [171, 134, 1], [109, 63, 54], [207, 138, 255], [151, 0, 95],
    [9, 80, 61], [84, 105, 51], [74, 65, 105], [166, 196, 102], [208, 195, 210],
    [255, 109, 65], [0, 143, 149], [179, 0, 194], [209, 99, 106], [5, 121, 0],
    [227, 255, 205], [147, 186, 208], [153, 69, 1], [3, 95, 161], [163, 255, 0],
    [119, 0, 170], [0, 182, 199], [0, 165, 120], [183, 130, 88], [95, 32, 0],
    [130, 114, 135], [110, 129, 133], [166, 74, 118], [219, 142, 185],
    [79, 210, 114], [178, 90, 62], [65, 70, 15], [127, 167, 115], [59, 105, 106],
    [142, 108, 45], [196, 172, 0], [95, 54, 80], [128, 76, 255], [201, 57, 1],
    [246, 0, 122], [191, 162, 208], [255, 255, 128], [147, 211, 203],
    [150, 100, 100], [168, 171, 172], [146, 112, 198], [210, 170, 100],
    [92, 136, 89], [218, 88, 184], [241, 129, 0], [217, 17, 255], [124, 74, 181],
    [70, 70, 70], [255, 228, 255], [154, 208, 0], [193, 0, 92], [76, 91, 113],
    [255, 180, 195], [106, 154, 176], [230, 150, 140], [60, 143, 255],
    [128, 64, 128], [92, 82, 55], [254, 212, 124], [73, 77, 174], [255, 160, 98],
    [255, 255, 255], [104, 84, 109], [169, 164, 131], [225, 199, 255],
    [137, 54, 74], [135, 158, 223], [7, 246, 231], [107, 255, 200], [58, 41, 149],
    [183, 121, 142], [255, 73, 97], [107, 142, 35], [190, 153, 153],
    [146, 139, 141], [70, 130, 180], [134, 199, 156], [209, 226, 140],
    [96, 36, 108], [96, 96, 96], [64, 170, 64], [152, 251, 152], [208, 229, 228],
    [206, 186, 171], [152, 161, 64], [116, 112, 0], [0, 114, 143], [102, 102, 156],
    [250, 141, 255], ]

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
        'n02691156': 0,
        'n02419796': 1,
        'n02131653': 2,
        'n02834778': 3,
        'n01503061': 4,
        'n02924116': 5,
        'n02958343': 6,
        'n02402425': 7,
        'n02084071': 8,
        'n02121808': 9,
        'n02503517': 10,
        'n02118333': 11,
        'n02510455': 12,
        'n02342885': 13,
        'n02374451': 14,
        'n02129165': 15,
        'n01674464': 16,
        'n02484322': 17,
        'n03790512': 18,
        'n02324045': 19,
        'n02509815': 20,
        'n02411705': 21,
        'n01726692': 22,
        'n02355227': 23,
        'n02129604': 24,
        'n04468005': 25,
        'n01662784': 26,
        'n04530566': 27,
        'n02062744': 28,
        'n02391049': 29}
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
        nm=obj.find('name').text
        # nm=vid_classes.code_to_class_string(str(nm.text))
        bb = obj.find('bndbox')
        gt = [0, 0, 0, 0]
        gt[0] = int(bb.find('xmin').text)
        gt[1] = int(bb.find('ymin').text)
        gt[2] = int(bb.find('xmax').text)
        gt[3] = int(bb.find('ymax').text)
        gt[2] = gt[2] - gt[0]
        gt[3] = gt[3] - gt[1]

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
    if det_or_vid == "VID":
        path_info = path_info[::5]
    path_info = [line.split(' ')[0] for line in path_info]
    dataset_dicts = []
    for idx, v in enumerate(path_info):
        record = {}

        filename = path_root+"Data/"+det_or_vid+"/"+train_or_val+"/"+v+".JPEG"
        gtfilepath= path_root+"Annotations/"+det_or_vid+"/"+train_or_val+"/"+v+".xml"
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
                "bbox_mode": BoxMode.XYWH_ABS,
                # "segmentation": [poly],
                "category_id": code_to_code_chall(str(img_info['name'][i])),
                # "iscrowd": 0
            }
            if obj["category_id"]=="nothing":
                # print(gtfilepath)
                continue
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def register_ILSVRC():
    for c in ["DET","VID"]:
    # for c in ["VID"]:
        for d in ["train", "val"]:
            DatasetCatalog.register("ILSVRC_" +c+"_"+ d, lambda d=d: get_ILSVRC_dicts(
                "/home/zb/project/detectron2/projects/adnet/datasets/data/ILSVRC/",("ImageSets/" +c+"/"+ d+".txt"),c,d))
            MetadataCatalog.get("ILSVRC_" +c+"_"+ d).set(thing_classes=CLASS_NAMES)
            MetadataCatalog.get("ILSVRC_" + c + "_" + d).set(thing_colors=OBJ_COLORS)


def testDataloader():
    for c in ["DET","VID"]:
    # for c in ["VID"]:
        for d in ["train", "val"]:
            pat = "/home/zb/project/detectron2/projects/adnet/datasets/tem/" + c + "/" + d + "/"
            if not os.path.exists(pat):
                os.makedirs(pat)
            dataset_dicts = get_ILSVRC_dicts(
                "/home/zb/project/detectron2/projects/adnet/datasets/data/ILSVRC/", ("ImageSets/" +c+"/"+ d+".txt"), c, d)
            ILSVRC_metadata = MetadataCatalog.get("ILSVRC_" +c+"_"+ d)
            for f,e in enumerate(random.sample(dataset_dicts, 3)):
                print(c,d,f)
                img = cv2.imread(e["file_name"])
                visualizer = Visualizer(img[:, :, ::-1], metadata=ILSVRC_metadata, scale=1)
                vis = visualizer.draw_dataset_dict(e)
                cv2_imshow(vis.get_image()[:, :, ::-1])
                filename=os.path.join(pat,e["file_name"][-10:])
                cv2.imwrite(filename, vis.get_image(), [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def trainILSVRC(args):
    register_ILSVRC()
    yaml_path,outdir,weights_name=get_cfg_info()
    cfg=setup(yaml_path,outdir,weights_name)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    return trainer.train()

def inferenceILSVRC(cfg):
    predictor = DefaultPredictor(cfg)
    dataset_dicts = get_ILSVRC_dicts(
        "/home/zb/project/detectron2/projects/adnet/datasets/data/ILSVRC/", ("ImageSets/" + "DET" + "/" + "val" + ".txt"), "DET", "val")
    ILSVRC_metadata = MetadataCatalog.get("ILSVRC_" +"DET"+"_"+ "val")
    pat = "/home/zb/project/detectron2/projects/adnet/datasets/tem/inferenceILSVRC/"
    if not os.path.exists(pat):
        os.makedirs(pat)
    for d in random.sample(dataset_dicts, 30):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=ILSVRC_metadata,
                       scale=1
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2_imshow(v.get_image()[:, :, ::-1])
        filename = os.path.join(pat, d["file_name"][-10:])
        cv2.imwrite(filename, v.get_image(), [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def evalILSVRC(cfg):
    # evaluator = COCOEvaluator("ILSVRC_DET_val", cfg, False, output_dir="tem/evalILSVRCoutput/")
    # val_loader = build_detection_test_loader(cfg, "ILSVRC_DET_val")
    # trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    # inference_on_dataset(trainer.model, val_loader, evaluator)
    pass

def setup(yaml_path,outdir,weights_name):
    cfg = get_cfg()
    cfg.merge_from_file(yaml_path)
    cfg.DATALOADER.NUM_WORKERS = 6
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 30  # only has one class (ballon)
    cfg.OUTPUT_DIR = outdir

    cfg.SOLVER.IMS_PER_BATCH = 10
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 600000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    # cfg.DATASETS.TRAIN = ("ILSVRC_DET_train", "ILSVRC_DET_val")
    cfg.DATASETS.TRAIN = ("ILSVRC_VID_train",)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, weights_name)  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.DATASETS.TEST = ("ILSVRC_DET_val",)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    return cfg

def train_main(args):
    print("Command Line Args:", args)
    num_gpus=2
    launch(
        trainILSVRC,
        num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

def get_cfg_info():
    yaml_path="/home/zb/project/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    outdir='/home/zb/project/detectron2/projects/adnet/datasets/tem/train_output/'
    # weights_name="model_final.pth"
    weights_name = "model_0299999.pth"
    return yaml_path,outdir,weights_name

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    train_main(args)  # trainILSVRC(cfg)

    # yaml_path,outdir,weights_name=get_cfg_info()
    # cfg=setup(yaml_path,outdir,weights_name)
    # register_ILSVRC()
    # testDataloader()
    # inferenceILSVRC(cfg)
    ## evalILSVRC(cfg) #not implemented
    print("finished.")