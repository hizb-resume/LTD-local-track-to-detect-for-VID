import _init_paths
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
from utils.my_util import get_ILSVRC_eval_infos,cal_iou,cal_success
from utils.overlap_ratio import overlap_ratio
from utils import vid_classes
import argparse
import copy
from prettytable import PrettyTable

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
    
parser = argparse.ArgumentParser(
    description='gen_gt_file')
parser.add_argument('--eval_imgs', default=0, type=int,
                    help='the num of imgs that picked from val.txt, 0 represent all imgs')
parser.add_argument('--gt_skip', default=5, type=int, help='frame sampling frequency')
parser.add_argument('--gengt', default=False, type=str2bool, help='generate gt results and save to file')
parser.add_argument('--doprecision', default=True, type=str2bool, help='run do precision function')
parser.add_argument('--evalfilepath', default='../datasets/data/ILSVRC-vid-eval-del-pred.txt', type=str, help='The eval results file')

def gen_gt_file(path,args):
    videos_infos,train_videos=get_ILSVRC_eval_infos(args)
    out_file = open('%s-gt.txt' % path, 'w')
    for tj in range(len(videos_infos)):
    # for tj in range(10):
        for ti in range(len(videos_infos[tj]['gt'])):
            for tk in range(len(videos_infos[tj]['gt'][ti])):
                # if tj==31 and ti==66:
                #     print("debug")
                out_file.write(str(tj) + ',' +
                               str(ti) + ',' +
                               str(videos_infos[tj]['trackid'][ti][tk]) + ',' +
                               str(videos_infos[tj]['name'][ti][tk]) + ',' +
                               str('1') + ',' +
                               str(videos_infos[tj]['gt'][ti][tk][0]) + ',' +
                               str(videos_infos[tj]['gt'][ti][tk][1]) + ',' +
                               str(videos_infos[tj]['gt'][ti][tk][2]) + ',' +
                               str(videos_infos[tj]['gt'][ti][tk][3]) + '\n')
    out_file.close()


def gen_pred_file(path,vid_pred):
    # out_file = open('%s-pred.txt' % path, 'w')
    out_file = open('%s-pred.txt' % path, 'a')
    for ti in range(len(vid_pred['bbox'])):
        # out_file.write(str(vid_pred['vid_id']) + ',' +
        #                str(vid_pred['frame_id'][ti]) + ',' +
        #                str(vid_pred['track_id'][ti]) + ',' +
        #                str(vid_pred['obj_name'][ti]) + ',' +
        #                str(vid_pred['score_cls'][ti]) + ',' +
        #                str(vid_pred['bbox'][ti][0]) + ',' +
        #                str(vid_pred['bbox'][ti][1]) + ',' +
        #                str(vid_pred['bbox'][ti][2]) + ',' +
        #                str(vid_pred['bbox'][ti][3]) + '\n')
        out_file.write(str(vid_pred['vid_id']) + ',' +
                       str(vid_pred['frame_id'][ti]) + ',' +
                       str(vid_pred['track_id'][ti]) + ',' +
                       str(vid_pred['detortrack'][ti]) + ',' +
                       str(vid_pred['obj_name'][ti]) + ',' +
                       '%.2f'%(vid_pred['score_cls'][ti]) + ',' +
                       '%.1f'%(vid_pred['bbox'][ti][0]) + ',' +
                       '%.1f'%(vid_pred['bbox'][ti][1]) + ',' +
                       '%.1f'%(vid_pred['bbox'][ti][2]) + ',' +
                       '%.1f'%(vid_pred['bbox'][ti][3]) + '\n')
    out_file.close()

def read_results_info(path_pred):
    vids_pred = []
    vid_pred = {
        'vid_id': 0,
        'frame_id': [],
        'track_id': [],
        'obj_name': [],
        'score_cls': [],
        'bbox': []
    }
    img_pred = {
        'track_id': [],
        'obj_name': [],
        'score_cls': [],
        'bbox': []
    }
    pred_file = open(path_pred, 'r')
    list1 = pred_file.readlines()
    # img_paths = [line.split(',') for line in list1]
    id_vid = -1
    id_frame = -1
    id_track = -1
    for line in list1:
        box_inf = []
        tsp = line.split(',')
        for ti in range(3):
            box_inf.append(int(tsp[ti]))
        box_inf.append(tsp[3])
        for ti in range(4, 9):
            box_inf.append(float(tsp[ti]))
        if id_vid == -1 and id_frame == -1 and id_track == -1:
            id_vid = box_inf[0]
            id_frame = box_inf[1]
            # id_track = box_inf[2]
            img_pred['track_id'].append(box_inf[2])
            img_pred['obj_name'].append(box_inf[3])
            img_pred['score_cls'].append(box_inf[4])
            img_pred['bbox'].append(box_inf[5:])
            vid_pred['vid_id'] = box_inf[0]
        else:
            if id_vid != box_inf[0]:
                vid_pred['frame_id'].append(id_frame)
                vid_pred['track_id'].append(img_pred['track_id'])
                vid_pred['obj_name'].append(img_pred['obj_name'])
                vid_pred['score_cls'].append(img_pred['score_cls'])
                vid_pred['bbox'].append(img_pred['bbox'])
                vids_pred.append(vid_pred)
                vid_pred = {
                    'vid_id': 0,
                    'frame_id': [],
                    'track_id': [],
                    'obj_name': [],
                    'score_cls': [],
                    'bbox': []
                }
                vid_pred['vid_id'] = box_inf[0]
                id_vid = box_inf[0]
                id_frame = box_inf[1]
                img_pred = {
                    'track_id': [],
                    'obj_name': [],
                    'score_cls': [],
                    'bbox': []
                }
                img_pred['track_id'].append(box_inf[2])
                img_pred['obj_name'].append(box_inf[3])
                img_pred['score_cls'].append(box_inf[4])
                img_pred['bbox'].append(box_inf[5:])
            else:
                if id_frame != box_inf[1]:
                    vid_pred['frame_id'].append(id_frame)
                    vid_pred['track_id'].append(img_pred['track_id'])
                    vid_pred['obj_name'].append(img_pred['obj_name'])
                    vid_pred['score_cls'].append(img_pred['score_cls'])
                    vid_pred['bbox'].append(img_pred['bbox'])
                    id_frame = box_inf[1]
                    img_pred = {
                        'track_id': [],
                        'obj_name': [],
                        'score_cls': [],
                        'bbox': []
                    }
                    img_pred['track_id'].append(box_inf[2])
                    img_pred['obj_name'].append(box_inf[3])
                    img_pred['score_cls'].append(box_inf[4])
                    img_pred['bbox'].append(box_inf[5:])
                else:
                    img_pred['track_id'].append(box_inf[2])
                    img_pred['obj_name'].append(box_inf[3])
                    img_pred['score_cls'].append(box_inf[4])
                    img_pred['bbox'].append(box_inf[5:])
    vid_pred['frame_id'].append(id_frame)
    vid_pred['track_id'].append(img_pred['track_id'])
    vid_pred['obj_name'].append(img_pred['obj_name'])
    vid_pred['score_cls'].append(img_pred['score_cls'])
    vid_pred['bbox'].append(img_pred['bbox'])
    vids_pred.append(vid_pred)
    # img_paths.append(box_inf)
    # img_paths=np.asarray(img_paths)
    # t1=img_paths[0]
    pred_file.close()
    return vids_pred


def read_pred_results_info(path_pred):
    vids_pred = []
    vid_pred = {
        'vid_id': 0,
        'frame_id': [],
        'track_id': [],
        'detortrack': [],
        'obj_name': [],
        'score_cls': [],
        'bbox': []
    }
    img_pred = {
        'track_id': [],
        'detortrack': [],
        'obj_name': [],
        'score_cls': [],
        'bbox': []
    }
    pred_file = open(path_pred, 'r')
    list1 = pred_file.readlines()
    # img_paths = [line.split(',') for line in list1]
    id_vid = -1
    id_frame = -1
    id_track = -1
    for line in list1:
        box_inf = []
        tsp = line.split(',')
        for ti in range(4):
            box_inf.append(int(tsp[ti]))
        box_inf.append(tsp[4])
        for ti in range(5, 10):
            box_inf.append(float(tsp[ti]))
        if id_vid == -1 and id_frame == -1 and id_track == -1:
            id_vid = box_inf[0]
            id_frame = box_inf[1]
            # id_track = box_inf[2]
            img_pred['track_id'].append(box_inf[2])
            img_pred['detortrack'].append(box_inf[3])
            img_pred['obj_name'].append(box_inf[4])
            img_pred['score_cls'].append(box_inf[5])
            img_pred['bbox'].append(box_inf[6:])
            vid_pred['vid_id'] = box_inf[0]
        else:
            if id_vid != box_inf[0]:
                vid_pred['frame_id'].append(id_frame)
                vid_pred['track_id'].append(img_pred['track_id'])
                vid_pred['detortrack'].append(img_pred['detortrack'])
                vid_pred['obj_name'].append(img_pred['obj_name'])
                vid_pred['score_cls'].append(img_pred['score_cls'])
                vid_pred['bbox'].append(img_pred['bbox'])
                vids_pred.append(vid_pred)
                vid_pred = {
                    'vid_id': 0,
                    'frame_id': [],
                    'track_id': [],
                    'detortrack': [],
                    'obj_name': [],
                    'score_cls': [],
                    'bbox': []
                }
                vid_pred['vid_id'] = box_inf[0]
                id_vid = box_inf[0]
                id_frame = box_inf[1]
                img_pred = {
                    'track_id': [],
                    'detortrack': [],
                    'obj_name': [],
                    'score_cls': [],
                    'bbox': []
                }
                img_pred['track_id'].append(box_inf[2])
                img_pred['detortrack'].append(box_inf[3])
                img_pred['obj_name'].append(box_inf[4])
                img_pred['score_cls'].append(box_inf[5])
                img_pred['bbox'].append(box_inf[6:])
            else:
                if id_frame != box_inf[1]:
                    vid_pred['frame_id'].append(id_frame)
                    vid_pred['track_id'].append(img_pred['track_id'])
                    vid_pred['detortrack'].append(img_pred['detortrack'])
                    vid_pred['obj_name'].append(img_pred['obj_name'])
                    vid_pred['score_cls'].append(img_pred['score_cls'])
                    vid_pred['bbox'].append(img_pred['bbox'])
                    id_frame = box_inf[1]
                    img_pred = {
                        'track_id': [],
                        'detortrack': [],
                        'obj_name': [],
                        'score_cls': [],
                        'bbox': []
                    }
                    img_pred['track_id'].append(box_inf[2])
                    img_pred['detortrack'].append(box_inf[3])
                    img_pred['obj_name'].append(box_inf[4])
                    img_pred['score_cls'].append(box_inf[5])
                    img_pred['bbox'].append(box_inf[6:])
                else:
                    img_pred['track_id'].append(box_inf[2])
                    img_pred['detortrack'].append(box_inf[3])
                    img_pred['obj_name'].append(box_inf[4])
                    img_pred['score_cls'].append(box_inf[5])
                    img_pred['bbox'].append(box_inf[6:])
    vid_pred['frame_id'].append(id_frame)
    vid_pred['track_id'].append(img_pred['track_id'])
    vid_pred['detortrack'].append(img_pred['detortrack'])
    vid_pred['obj_name'].append(img_pred['obj_name'])
    vid_pred['score_cls'].append(img_pred['score_cls'])
    vid_pred['bbox'].append(img_pred['bbox'])
    vids_pred.append(vid_pred)
    # img_paths.append(box_inf)
    # img_paths=np.asarray(img_paths)
    # t1=img_paths[0]
    pred_file.close()
    return vids_pred

def maxiou(box,bboxs_pred):
    if len(bboxs_pred)==0:
        return 0,0
    max_id=0
    max_iou=0
    for t_id,bbox_pred in enumerate(bboxs_pred):
        iou=cal_iou(box,bbox_pred)
        if iou>max_iou:
            max_iou=iou
            max_id=t_id
    return max_id,max_iou

def do_precison(path_pred,path_gt):
    #correct ratio in predicted result
    ious=[]
    ious_cls=[]
    vids_pred =read_results_info(path_pred)
    vids_gt =read_results_info(path_gt)
    j = 0
    for i in range(len(vids_pred)):
        idv=vids_pred[i]['vid_id']
        while idv!=vids_gt[j]['vid_id']:
            # print("i_pred: %d, vid_id_pred: %d; j_gt: %d, vid_id_gt: %d."%(i,idv,j,vids_gt[j]['vid_id']))
            j+=1
        l=0
        for k in range(len(vids_pred[i]['frame_id'])):
            idf=vids_pred[i]['frame_id'][k]
            while idf !=vids_gt[j]['frame_id'][l]:
                l+=1
            bboxs_gt=vids_gt[j]['bbox'][l]
            bboxs_pred= vids_pred[i]['bbox'][k]
            for id_bgt,box in enumerate(bboxs_gt):
                id_iou,iou=maxiou(box,bboxs_pred)
                ious.append(iou)
                if vids_pred[i]['obj_name'][k][id_iou]==vids_gt[j]['obj_name'][l][id_bgt]:
                    ious_cls.append(iou)
                else:
                    ious_cls.append(0)
            l += 1
        j+=1
    iou_success_all=cal_success(ious)
    cls_success_all = cal_success(ious_cls)
    print('iou precision(iou>%.2f): %.2f'%(iou_success_all[14][0],iou_success_all[14][1]))
    print('cls precision(iou>%.2f): %.2f'%(cls_success_all[14][0],cls_success_all[14][1]))

def do_precison2(path_pred,path_gt):
    CLASS_NAMES = [
        'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
        'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda',
        'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle',
        'rabbit', 'red_panda', 'sheep', 'snake', 'squirrel', 'tiger',
        'train', 'turtle', 'watercraft', 'whale', 'zebra'
    ]
    ious=[]
    ious_cls=[]
    # vids_pred =read_results_info(path_pred)
    vids_pred = read_pred_results_info(path_pred)
    vids_gt =read_results_info(path_gt)
    n_all_boxes=0
    n_miss_boxes=0
    n_all_pics=0
    n_miss_pics=0
    cls_info={
        "name":"",
        "n_instances":0,
        "n_missed":0,
        "n_track": 0,
        "ious":[],
        "ious_cls":[],
        "iou_success_all": [],
        "cls_success_all":[]
    }
    total_inf=[]
    for ito in range(len(CLASS_NAMES)):
        total_inf.append(copy.deepcopy(cls_info))
        total_inf[ito]["name"]=CLASS_NAMES[ito]
    i = 0
    for j in range(len(vids_gt)):
        idg=vids_gt[j]['vid_id']
        if i>=len(vids_pred):
            for l in range(len(vids_gt[j]['frame_id'])):
                bboxs_gt = vids_gt[j]['bbox'][l]
                for tid in range(len(bboxs_gt)):
                    cls_name=vids_gt[j]['obj_name'][l][tid]
                    cls_id=int(vid_classes.class_string_to_comp_code(str(cls_name)))-1
                    total_inf[cls_id]["ious"].append(0)
                    total_inf[cls_id]["ious_cls"].append(0)
                    total_inf[cls_id]["n_instances"]+=1
                    total_inf[cls_id]["n_missed"]+=1
                    ious.append(0)
                    ious_cls.append(0)
                    n_all_boxes+=1
                    n_miss_boxes+=1
                if len(bboxs_gt)>0:
                    n_all_pics+=1
                    n_miss_pics+=1
        elif idg!=vids_pred[i]['vid_id']:
            # print("i_pred: %d, vid_id_pred: %d; j_gt: %d, vid_id_gt: %d."%(i,idv,j,vids_gt[j]['vid_id']))
            if idg<vids_pred[i]['vid_id']:
                for l in range(len(vids_gt[j]['frame_id'])):
                    bboxs_gt = vids_gt[j]['bbox'][l]
                    for tid in range(len(bboxs_gt)):
                        cls_name = vids_gt[j]['obj_name'][l][tid]
                        cls_id = int(vid_classes.class_string_to_comp_code(str(cls_name))) - 1
                        total_inf[cls_id]["ious"].append(0)
                        total_inf[cls_id]["ious_cls"].append(0)
                        total_inf[cls_id]["n_instances"] += 1
                        total_inf[cls_id]["n_missed"] += 1
                        ious.append(0)
                        ious_cls.append(0)
                        n_all_boxes += 1
                        n_miss_boxes += 1
                    if len(bboxs_gt) > 0:
                        n_all_pics += 1
                        n_miss_pics += 1
            else:
                print("test, this situation1 is not possible.")
        else:
            k=0
            for l in range(len(vids_gt[j]['frame_id'])):
                idgf=vids_gt[j]['frame_id'][l]
                if k>=len(vids_pred[i]['frame_id']):
                    bboxs_gt = vids_gt[j]['bbox'][l]
                    for tid in range(len(bboxs_gt)):
                        cls_name = vids_gt[j]['obj_name'][l][tid]
                        cls_id = int(vid_classes.class_string_to_comp_code(str(cls_name))) - 1
                        total_inf[cls_id]["ious"].append(0)
                        total_inf[cls_id]["ious_cls"].append(0)
                        total_inf[cls_id]["n_instances"] += 1
                        total_inf[cls_id]["n_missed"] += 1
                        ious.append(0)
                        ious_cls.append(0)
                        n_all_boxes += 1
                        n_miss_boxes += 1
                    if len(bboxs_gt) > 0:
                        n_all_pics += 1
                        n_miss_pics += 1
                elif idgf !=vids_pred[i]['frame_id'][k]:
                    if idgf < vids_pred[i]['frame_id'][k]:
                        bboxs_gt = vids_gt[j]['bbox'][l]
                        for tid in range(len(bboxs_gt)):
                            cls_name = vids_gt[j]['obj_name'][l][tid]
                            cls_id = int(vid_classes.class_string_to_comp_code(str(cls_name))) - 1
                            total_inf[cls_id]["ious"].append(0)
                            total_inf[cls_id]["ious_cls"].append(0)
                            total_inf[cls_id]["n_instances"] += 1
                            total_inf[cls_id]["n_missed"] += 1
                            ious.append(0)
                            ious_cls.append(0)
                            n_all_boxes += 1
                            n_miss_boxes += 1
                        if len(bboxs_gt) > 0:
                            n_all_pics += 1
                            n_miss_pics += 1
                    else:
                        print("test, this situation2 is not possible.")
                else:
                    bboxs_gt=vids_gt[j]['bbox'][l]
                    bboxs_pred= vids_pred[i]['bbox'][k]
                    for id_bgt,box in enumerate(bboxs_gt):
                        id_iou,iou=maxiou(box,bboxs_pred)
                        cls_name = vids_gt[j]['obj_name'][l][id_bgt]
                        cls_id = int(vid_classes.class_string_to_comp_code(str(cls_name))) - 1
                        total_inf[cls_id]["ious"].append(iou)
                        total_inf[cls_id]["n_instances"] += 1
                        trackid1=vids_gt[j]['track_id'][l][id_bgt]
                        for tck2 in range(len(bboxs_pred)):
                            trackid2=vids_pred[i]['track_id'][k][tck2]
                            if trackid1==trackid2:
                                total_inf[cls_id]["n_track"] += vids_pred[i]['detortrack'][k][id_bgt]
                        ious.append(iou)
                        if vids_pred[i]['obj_name'][k][id_iou]==vids_gt[j]['obj_name'][l][id_bgt]:
                            total_inf[cls_id]["ious_cls"].append(iou)
                            ious_cls.append(iou)
                        else:
                            total_inf[cls_id]["ious_cls"].append(0)
                            # total_inf[cls_id]["n_missed"] += 1
                            ious_cls.append(0)
                            # print("gt: %s, pred: %s"%(vids_gt[j]['obj_name'][l][id_bgt],vids_pred[i]['obj_name'][k][id_iou]))
                        n_all_boxes += 1
                        # n_miss_boxes += 1
                    if len(bboxs_gt) > 0:
                        n_all_pics += 1
                        # n_miss_pics += 1
                    k += 1
            i+=1
    rltTable = PrettyTable(["category", "n_box", "n_miss","miss_ratio", "n_track", "AP50_iou",
                            "AP50_cls", "AP60_iou", "AP60_cls", "AP70_iou", "AP70_cls"])
    totalRow=copy.deepcopy(cls_info)
    totalRow["name"]="Total"
    for ito in range(len(CLASS_NAMES)):
        total_inf[ito]["iou_success_all"] =cal_success(total_inf[ito]["ious"])
        total_inf[ito]["cls_success_all"] =cal_success(total_inf[ito]["ious_cls"])
        if total_inf[ito]["n_instances"]==0:
            ttem='0 %'
        else:
            ttem=str(round(total_inf[ito]["n_missed"]/total_inf[ito]["n_instances"]*100,2))+'%'
        rltTable.add_row([total_inf[ito]["name"],total_inf[ito]["n_instances"],total_inf[ito]["n_missed"],
                          ttem,
                          total_inf[ito]["n_track"],
                          total_inf[ito]["iou_success_all"][10][1],total_inf[ito]["cls_success_all"][10][1],
                          total_inf[ito]["iou_success_all"][12][1], total_inf[ito]["cls_success_all"][12][1],
                          total_inf[ito]["iou_success_all"][14][1], total_inf[ito]["cls_success_all"][14][1],])
        totalRow["n_instances"]+=total_inf[ito]["n_instances"]
        totalRow["n_missed"] += total_inf[ito]["n_missed"]
        totalRow["n_track"] += total_inf[ito]["n_track"]
        totalRow["ious"].extend(total_inf[ito]["ious"])
        totalRow["ious_cls"].extend(total_inf[ito]["ious_cls"])
    totalRow["iou_success_all"]=cal_success(totalRow["ious"])
    totalRow["cls_success_all"] = cal_success(totalRow["ious_cls"])
    rltTable.add_row(["----------","------","-----","------","-------","------","-------","-------","-------","-------","-------",])
    if total_inf[ito]["n_instances"]==0:
        ttem='0 %'
    else:
        ttem=str(round(totalRow["n_missed"]/totalRow["n_instances"]*100,2))+'%'
    rltTable.add_row([totalRow["name"], totalRow["n_instances"], totalRow["n_missed"],
                      ttem,
                      totalRow["n_track"],
                      totalRow["iou_success_all"][10][1], totalRow["cls_success_all"][10][1],
                      totalRow["iou_success_all"][12][1], totalRow["cls_success_all"][12][1],
                      totalRow["iou_success_all"][14][1], totalRow["cls_success_all"][14][1], ])
    rltTable.align["n_instances"] = "l"
    print(rltTable)
    # iou_success_all=cal_success(ious)
    # cls_success_all = cal_success(ious_cls)
    # print('iou precision(iou>%.2f): %.2f%%.' % (iou_success_all[10][0], (iou_success_all[10][1]) * 100))
    # print('cls precision(iou>%.2f): %.2f%%.' % (cls_success_all[10][0], (cls_success_all[10][1]) * 100))
    # print('iou precision(iou>%.2f): %.2f%%.' % (iou_success_all[12][0], (iou_success_all[12][1]) * 100))
    # print('cls precision(iou>%.2f): %.2f%%.' % (cls_success_all[12][0], (cls_success_all[12][1]) * 100))
    # print('iou precision(iou>%.2f): %.2f%%.'%(iou_success_all[14][0],(iou_success_all[14][1])*100))
    # print('cls precision(iou>%.2f): %.2f%%.'%(cls_success_all[14][0],(cls_success_all[14][1])*100))
    # print('all gt imgs: %d, missed imgs: %d, missed img ratio: %.2f%%.'%(n_all_pics,n_miss_pics,(n_miss_pics/n_all_pics)*100))
    # print('all gt boxes: %d, missed boxess: %d, missed box ratio: %.2f%%.'%(n_all_boxes,n_miss_boxes,(n_miss_boxes/n_all_boxes)*100))

if __name__ == "__main__":
    args = parser.parse_args()
    if args.gengt:
        gen_gt_file('../datasets/data/ILSVRC-vid-eval',args)
    if args.doprecision:
        do_precison2(args.evalfilepath,'../datasets/data/ILSVRC-vid-eval-gt.txt')