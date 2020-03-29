#ADNet/adnet_test.m
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
import numpy as np
from PIL import Image #use PIL to processs img
import torch
import torchsnooper
import torch.optim as optim
from models.ADNet import adnet
from options.general import opts
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import glob
from datasets.online_adaptation_dataset import OnlineAdaptationDataset, OnlineAdaptationDatasetStorage
from utils.augmentations import ADNet_Augmentation,ADNet_Augmentation2,ADNet_Augmentation3
from utils.do_action import do_action
import time
from utils.display import display_result, draw_boxes
from utils.gen_samples import gen_samples
from utils.precision_plot import distance_precision_plot, iou_precision_plot
from utils.my_util import aHash,Hamming_distance,cal_iou
from random import shuffle
from tensorboardX import SummaryWriter
from detectron2.structures import Boxes,RotatedBoxes
from detectron2.utils.visualizer import Visualizer

def pred(predictor,class_names,frame):
    outputs = predictor(frame)
    predictions = outputs["instances"].to("cpu")
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes if predictions.has("pred_classes") else None
    labels = [class_names[i] for i in classes]
    if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
        boxes = boxes.tensor.numpy()
    else:
        boxes = np.asarray(boxes)
    num_instances = len(boxes)
    if num_instances != 0:
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    scores = scores.numpy()
    assert len(scores) == num_instances

    classes = classes.numpy()
    assert len(classes) == num_instances

    return boxes,labels,scores

# def _create_text_labels(classes, scores, class_names):
#     """
#     Args:
#         classes (list[int] or None):
#         scores (list[float] or None):
#         class_names (list[str] or None):
#
#     Returns:
#         list[str] or None
#     """
#     labels = None
#     if classes is not None and class_names is not None and len(class_names) > 1:
#         labels = [class_names[i] for i in classes]
#     if scores is not None:
#         if labels is None:
#             labels = ["{:.0f}%".format(s * 100) for s in scores]
#         else:
#             labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
#     return labels

# @torchsnooper.snoop()
def adnet_test(net, predictor,siamesenet,metalog,class_names,vidx,vid_path, opts, args):

    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print(
                "WARNING: It looks like you have a CUDA device, but aren't " + "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # transform = ADNet_Augmentation(opts)

    mean = np.array(opts['means'], dtype=np.float32)
    mean = torch.from_numpy(mean).cuda()
    transform = ADNet_Augmentation2(opts,mean)

    transform3_adition = transforms.Compose([transforms.Resize((100, 100)),
                                    transforms.ToTensor()
                                    ])
    transform3 = ADNet_Augmentation3(transform3_adition)

    if isinstance(vid_path,list):
        print('Testing sequences in ' + str(vid_path[0][-43:-12]) + '...')
    else:
        print('Testing sequences in ' + str(vid_path) + '...')
    t_sum = 0

    if args.visualize:
        writer = SummaryWriter(log_dir=os.path.join('tensorboardx_log', 'online_adapatation_' + args.save_result_npy))

    ################################
    # Load video sequences
    ################################

    vid_info={
        'gt' : [],
        'img_files' : [],
        'nframes' : 0
    }

    vid_gt={
        'vid_id':vidx,
        'frame_id':[],
        'track_id':[],
        'obj_name':[],
        'score_cls':[],
        'bbox':[]
    }
    vid_pred={
        'vid_id':vidx,
        'frame_id':[],
        'track_id':[],
        'detortrack':[],    #0 means det, 1 means track
        'obj_name':[],
        'score_cls':[],
        'bbox':[]
    }

    frame_pred = {
        'frame_id': 0,
        'track_id': [],
        'obj_name': [],
        'score_cls': [],
        'bbox': []
    }

    pre_aera=[]
    curr_aera=[]

    spend_time={
        'predict':0,
        'n_predict_frames':0,
        'track':0,
        'n_track_frames':0,
        'readframe':0,
        'n_readframe':0,
        'append':0,
        'n_append':0,
        'transform':0,
        'n_transform':0,
        # 'cuda':0,
        # 'n_cuda':0,
        'argmax_after_forward':0,
        'n_argmax_after_forward':0,
        'do_action':0,
        'n_do_action':0
    }

    #vid_info['img_files'] = glob.glob(os.path.join(vid_path, 'img', '*.jpg'))
    #vid_info['img_files'] = glob.glob(os.path.join(vid_path, '*.jpg'))
    isVidFile=False
    if isinstance(vid_path,list):
        vid_info['img_files'] =vid_path
    else:
        if '.' in vid_path[-5:]:
            isVidFile = True
        else:
            vid_info['img_files'] = glob.glob(os.path.join(vid_path, '*.JPEG'))
            vid_info['img_files'].sort(key=str.lower)

    #gt_path = os.path.join(vid_path, 'groundtruth_rect.txt')
    # gt_path = os.path.join(vid_path, 'groundtruth.txt')
    #
    # if not os.path.exists(gt_path):
    #     bboxes = []
    #     t = 0
    #     return bboxes, t_sum
    #
    # # parse gt
    # gtFile = open(gt_path, 'r')
    # gt = gtFile.read().split('\n')
    # for i in range(len(gt)):
    #     if gt[i] == '' or gt[i] is None:
    #         continue
    #
    #     if ',' in gt[i]:
    #         separator = ','
    #     elif '\t' in gt[i]:
    #         separator = '\t'
    #     elif ' ' in gt[i]:
    #         separator = ' '
    #     else:
    #         separator = ','
    #
    #     gt[i] = gt[i].split(separator)
    #     gt[i] = list(map(float, gt[i]))
    # gtFile.close()
    #
    # if len(gt[0]) >= 6:
    #     for gtidx in range(len(gt)):
    #         if gt[gtidx] == "":
    #             continue
    #         x = gt[gtidx][0:len(gt[gtidx]):2]
    #         y = gt[gtidx][1:len(gt[gtidx]):2]
    #         gt[gtidx] = [min(x),
    #                      min(y),
    #                      max(x) - min(x),
    #                      max(y) - min(y)]
    #
    # vid_info['gt'] = gt
    # if vid_info['gt'][-1] == '':  # small hack
    #     vid_info['gt'] = vid_info['gt'][:-1]
    # vid_info['nframes'] = min(len(vid_info['img_files']), len(vid_info['gt']))
    cap=None
    if isVidFile == True:
        cap = cv2.VideoCapture(vid_path)
        vid_info['nframes'] =int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # vid_info['nframes'] = 15
    else:
        # vid_info['nframes'] =48
        vid_info['nframes'] = len(vid_info['img_files'])
    # catch the first box
    # curr_bbox = vid_info['gt'][0]
    # curr_bbox = [114,158,88,100]

    

    # init containers
    # bboxes = np.zeros(np.array(vid_info['gt']).shape)  # tracking result containers

    ntraining = 0

    # setup training
    if args.cuda:
        optimizer = optim.SGD([
            {'params': net.module.base_network.parameters(), 'lr': 0},
            {'params': net.module.fc4_5.parameters()},
            {'params': net.module.fc6.parameters()},
            {'params': net.module.fc7.parameters(), 'lr': 1e-3}],
            lr=1e-3, momentum=opts['train']['momentum'], weight_decay=opts['train']['weightDecay'])
    else:
        optimizer = optim.SGD([
            {'params': net.base_network.parameters(), 'lr': 0},
            {'params': net.fc4_5.parameters()},
            {'params': net.fc6.parameters()},
            {'params': net.fc7.parameters(), 'lr': 1e-3}],
            lr=1e-3, momentum=opts['train']['momentum'], weight_decay=opts['train']['weightDecay'])

    action_criterion = nn.CrossEntropyLoss()
    score_criterion = nn.CrossEntropyLoss()

    dataset_storage_pos = None
    dataset_storage_neg = None
    is_negative = False  # is_negative = True if the tracking failed
    target_score = 0
    all_iteration = 0
    t = 0

    # vidpath = "../../../demo/examples/jiaotong2.avi"
    # cap = cv2.VideoCapture(vidpath)
    # length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # for frame_idx in range(length):
    sign_redet=False
    dis_redet=0
    n_trackid=-1
    obj_area=[]
    obj_box=[]
    siam_thred_inf=[]
    for frame_idx in range(vid_info['nframes']):
    ## for frame_idx, frame_path in enumerate(vid_info['img_files']):
        # frame_idx = idx
        ts1=time.time()
        if isVidFile == True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
            success, frame = cap.read()
        else:
            frame_path = vid_info['img_files'][frame_idx]
            frame = cv2.imread(frame_path)
            try:
                frame.shape
            except:
                print(frame_path)
        ts2=time.time()
        spend_time['readframe'] += ts2 - ts1
        spend_time['n_readframe'] += 1

        t0_wholetracking = time.time()

        # cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        # success, frame = cap.read()
        if len(frame_pred['bbox']) == 0:
            sign_redet = True
            # print('the num of pred boxes is 0! pre frame: %d, now frame: %d .'%(frame_idx-1,frame_idx))
        if frame_idx==0 or sign_redet==True or dis_redet==20:
        # if frame_idx == 0 or sign_redet == True:
            # print('redetection: frame %d'%frame_idx)
            ts1=time.time()
            boxes,classes,scores = pred(predictor,class_names, frame)
            ts2=time.time()
            spend_time['predict']+=ts2-ts1
            spend_time['n_predict_frames']+=1
            frame_pred['frame_id'] = frame_idx
            frame_pred['track_id'] = []
            frame_pred['obj_name'] = []
            frame_pred['bbox'] = []
            frame_pred['score_cls'] = []
            pre_aera=[]
            pre_aera_crop = []
            ts1=time.time()
            n_bbox=len(boxes)
            # if frame_idx==0:
            if n_trackid==-1:
                for i_d in range(n_bbox):
                    n_trackid+=1
                    frame_pred['track_id'].append(n_trackid)
                    frame_pred['obj_name'].append(classes[i_d])
                    frame_pred['bbox'].append(boxes[i_d])
                    frame_pred['score_cls'].append(scores[i_d])
                    if args.useSiamese or args.checktrackid:
                        siam_thred_inf.append(0)
                        t_aera, t_aera_crop,_ = transform3(frame, boxes[i_d])
                        if args.useSiamese:
                            pre_aera.append(t_aera)
                            pre_aera_crop.append(t_aera_crop)
                        if args.checktrackid:
                            obj_area.append(t_aera)
                            obj_box.append(boxes[i_d])
            else:
                siam_thred_inf = []
                for i_d in range(n_bbox):
                    if args.useSiamese or args.checktrackid:
                        t_aera, t_aera_crop,_ = transform3(frame, boxes[i_d])
                        if args.useSiamese:
                            pre_aera.append(t_aera)
                            pre_aera_crop.append(t_aera_crop)
                        if args.checktrackid:
                            # calculate the id of the box with the highest similarity
                            # if thred of the highest similarity is higher than 0.9,
                            # n_trackid +1
                            maxid=0
                            maxdistance=9999
                            for nt in range(len(obj_area)):
                                output1, output2 = siamesenet(Variable(t_aera).cuda(), Variable(obj_area[nt]).cuda())
                                euclidean_distance = F.pairwise_distance(output1, output2)
                                if euclidean_distance.item()<maxdistance:
                                    maxdistance=euclidean_distance.item()
                                    maxid=nt
                            # print(maxid,len(obj_box))
                            siam_thred_inf.append(maxdistance)
                            if maxdistance>args.siam_thred and cal_iou(boxes[i_d],obj_box[maxid])<0.6:
                                n_trackid += 1
                                obj_area.append(t_aera)
                                obj_box.append(boxes[i_d])
                                frame_pred['track_id'].append(n_trackid)
                            else:
                                frame_pred['track_id'].append(maxid)
                                obj_box[maxid]=boxes[i_d]
                        else:
                            # n_trackid_t=i_d
                            frame_pred['track_id'].append(i_d)
                            siam_thred_inf.append(0)
                            # obj_area.append(t_aera)
                    else:
                        frame_pred['track_id'].append(i_d)
                    # n_trackid+=1
                    # frame_pred['track_id'].append(n_trackid_t)
                    frame_pred['obj_name'].append(classes[i_d])
                    frame_pred['bbox'].append(boxes[i_d])
                    frame_pred['score_cls'].append(scores[i_d])

            vid_pred['frame_id'].extend(np.full(n_bbox, frame_pred['frame_id']))
            vid_pred['track_id'].extend(frame_pred['track_id'])
            vid_pred['detortrack'].extend(np.full(n_bbox,0))
            vid_pred['obj_name'].extend(frame_pred['obj_name'])
            vid_pred['bbox'].extend(frame_pred['bbox'])
            vid_pred['score_cls'].extend(frame_pred['score_cls'])
            if args.track:
                sign_redet = False
            dis_redet = 0
            ts2=time.time()
            spend_time['append'] += ts2 - ts1
            spend_time['n_append'] += 1

            # curr_bbox = boxes[2]

        # draw box or with display, then save
        # if args.display_images:
        #     im_with_bb = display_result(frame, frame_pred['bbox'])  # draw box and display
        # else:
        #     im_with_bb = draw_boxes(frame, frame_pred['bbox'])

        # if args.save_result_images:
        #     filename = os.path.join(args.save_result_images, str(frame_idx).rjust(4,'0')+'-00-00-patch_initial.jpg')
        #     cv2.imwrite(filename, im_with_bb)

        # curr_bbox_old = curr_bbox
        # cont_negatives = 0

        # if frame_idx > 0:
        else:
            dis_redet += 1
            # tracking
            # if args.cuda:
            #     net.module.set_phase('test')
            # else:
            #     net.set_phase('test')
            frame_pred['frame_id'] = frame_idx
            # if len(frame_pred['bbox'])==0:
            #     sign_redet=True
            #     continue
            #     print('the num of pred boxes is 0!')

            ts_all = 0
            frame2=frame.copy()
            frame2=frame2.astype(np.float32)
            frame2=torch.from_numpy(frame2).cuda()
            siam_thred_inf = []
            for t_id,curr_bbox in enumerate(frame_pred['bbox']):
                t = 0
                while True:
                    ts1=time.time()
                    # curr_patch, curr_bbox, _, _ = transform(frame, curr_bbox, None, None)
                    curr_patch, curr_bbox, _, _ = transform(frame2, curr_bbox, None, None)
                    ts2=time.time()
                    spend_time['transform'] += ts2 - ts1
                    spend_time['n_transform'] += 1
                    # ts1 = time.time()
                    # if args.cuda:
                    #     curr_patch = curr_patch.cuda()  #this step need most of the time
                    # ts2 = time.time()
                    # spend_time['cuda'] += ts2 - ts1
                    # spend_time['n_cuda'] += 1
                    # curr_patch = curr_patch.unsqueeze(0)  # 1 batch input [1, curr_patch.shape]
                    ts1 = time.time()
                    fc6_out, fc7_out = net.forward(curr_patch)
                    ts2 = time.time()
                    ts_all+=ts2-ts1
                    ts1=time.time()
                    curr_score = fc7_out.detach().cpu().numpy()[0][1]

                    # print(curr_score)

                    # if ntraining > args.believe_score_result:
                    #     if curr_score < opts['failedThre']:
                    #         cont_negatives += 1

                    if args.cuda:
                        action = np.argmax(fc6_out.detach().cpu().numpy())  # TODO: really okay to detach?
                        action_prob = fc6_out.detach().cpu().numpy()[0][action]
                    else:
                        action = np.argmax(fc6_out.detach().numpy())  # TODO: really okay to detach?
                        action_prob = fc6_out.detach().numpy()[0][action]
                    ts2=time.time()
                    spend_time['argmax_after_forward'] += ts2 - ts1
                    spend_time['n_argmax_after_forward'] += 1
                    # do action
                    ts1 = time.time()
                    curr_bbox = do_action(curr_bbox, opts, action, frame.shape)

                    # bound the curr_bbox size
                    if curr_bbox[2] < 10:
                        curr_bbox[0] = min(0, curr_bbox[0] + curr_bbox[2] / 2 - 10 / 2)
                        curr_bbox[2] = 10
                    if curr_bbox[3] < 10:
                        curr_bbox[1] = min(0, curr_bbox[1] + curr_bbox[3] / 2 - 10 / 2)
                        curr_bbox[3] = 10
                    ts2 = time.time()
                    spend_time['do_action'] += ts2 - ts1
                    spend_time['n_do_action'] += 1

                    t += 1

                    # draw box or with display, then save
                    if args.display_images_t:
                        im_with_bb = display_result(frame, curr_bbox)  # draw box and display
                    else:
                        im_with_bb = draw_boxes(frame, curr_bbox)

                    if args.save_result_images_t:
                        filename = os.path.join(args.save_result_images, str(frame_idx).rjust(4,'0')+'-'+str(t_id).rjust(2,'0')+'-' + str(t).rjust(2,'0') + '.jpg')
                        cv2.imwrite(filename, im_with_bb)
                        pass

                    if action == opts['stop_action'] or t >= opts['num_action_step_max']:
                        break

                # print('final curr_score: %.4f' % curr_score)

                # redetection when confidence < threshold 0.5. But when fc7 is already reliable. Else, just trust the ADNet
                # if ntraining > args.believe_score_result:

                if curr_score < 0.5:
                    # print('redetection: frame %d' % frame_idx)
                    is_negative = True
                    dis_redet = 0
                    ts1=time.time()
                    boxes, classes, scores = pred(predictor, class_names,frame)
                    ts2=time.time()
                    spend_time['predict'] += ts2 - ts1
                    spend_time['n_predict_frames'] += 1
                    # frame_pred['frame_id'] = frame_idx
                    frame_pred['track_id']=[]
                    frame_pred['obj_name']=[]
                    frame_pred['bbox']=[]
                    frame_pred['score_cls']=[]
                    pre_aera=[]
                    pre_aera_crop=[]
                    ts1=time.time()
                    n_bbox = len(boxes)
                    siam_thred_inf=[]
                    for i_d in range(n_bbox):
                        if args.useSiamese or args.checktrackid:
                            t_aera, t_aera_crop, _ = transform3(frame, boxes[i_d])
                            if args.useSiamese:
                                pre_aera.append(t_aera)
                                pre_aera_crop.append(t_aera_crop)
                            if args.checktrackid:
                                # calculate the id of the box with the highest similarity
                                # if thred of the highest similarity is higher than 0.9,
                                # n_trackid +1
                                maxid = 0
                                maxdistance = 9999
                                for nt in range(len(obj_area)):
                                    output1, output2 = siamesenet(Variable(t_aera).cuda(),
                                                                  Variable(obj_area[nt]).cuda())
                                    euclidean_distance = F.pairwise_distance(output1, output2)
                                    if euclidean_distance.item() < maxdistance:
                                        maxdistance = euclidean_distance.item()
                                        maxid = nt
                                siam_thred_inf.append(maxdistance)
                                if maxdistance > args.siam_thred and cal_iou(boxes[i_d], obj_box[maxid]) < 0.6:
                                    n_trackid += 1
                                    obj_area.append(t_aera)
                                    obj_box.append(boxes[i_d])
                                    frame_pred['track_id'].append(n_trackid)
                                else:
                                    frame_pred['track_id'].append(maxid)
                                    obj_box[maxid] = boxes[i_d]
                            else:
                                # n_trackid_t=i_d
                                frame_pred['track_id'].append(i_d)
                                siam_thred_inf.append(0)
                                # obj_area.append(t_aera)
                        else:
                            frame_pred['track_id'].append(i_d)
                        frame_pred['obj_name'].append(classes[i_d])
                        frame_pred['bbox'].append(boxes[i_d])
                        frame_pred['score_cls'].append(scores[i_d])

                    ts2=time.time()
                    spend_time['append'] += ts2 - ts1
                    # vid_pred['frame_id'].extend(np.full(n_bbox, frame_pred['frame_id']))
                    # vid_pred['track_id'].extend(frame_pred['track_id'])
                    # vid_pred['obj_name'].extend(frame_pred['obj_name'])
                    # vid_pred['bbox'].extend(frame_pred['bbox'])
                    # vid_pred['score_cls'].extend(frame_pred['score_cls'])

                    # redetection process
                    # redet_samples = gen_samples('gaussian', curr_bbox_old, opts['redet_samples'], opts, min(1.5, 0.6 * 1.15 ** cont_negatives), opts['redet_scale_factor'])
                    # score_samples = []
                    #
                    # for redet_sample in redet_samples:
                    #     temp_patch, temp_bbox, _, _ = transform(frame, redet_sample, None, None)
                    #     if args.cuda:
                    #         temp_patch = temp_patch.cuda()
                    #
                    #     temp_patch = temp_patch.unsqueeze(0)  # 1 batch input [1, curr_patch.shape]
                    #
                    #     fc6_out_temp, fc7_out_temp = net.forward(temp_patch)
                    #
                    #     score_samples.append(fc7_out_temp.detach().cpu().numpy()[0][1])
                    #
                    # score_samples = np.array(score_samples)
                    # max_score_samples_idx = np.argmax(score_samples)
                    #
                    # # replace the curr_box with the samples with maximum score
                    # curr_bbox = redet_samples[max_score_samples_idx]
                    #
                    # update the final result image
                    # if args.display_images:
                    #     im_with_bb = display_result(frame, frame_pred['bbox'])  # draw box and display
                    # else:
                    #     im_with_bb = draw_boxes(frame, frame_pred['bbox'])
                    #
                    # if args.save_result_images:
                    #     filename = os.path.join(args.save_result_images, str(frame_idx).rjust(4,'0') + '-98-20-redet.jpg')
                    #     cv2.imwrite(filename, im_with_bb)
                    break
                else:
                    is_negative = False
                    # frame_pred['frame_id'] = frame_idx
                    # frame_pred['track_id'] = []
                    # frame_pred['obj_name'] = []
                    frame_pred['bbox'][t_id] = curr_bbox
                    frame_pred['score_cls'][t_id] = curr_score

                    if args.useSiamese:
                        curr_aera, curr_aera_crop, _ = transform3(frame, curr_bbox)
                        x0=pre_aera[t_id]
                        x0_crop=pre_aera_crop[t_id]
                        output1, output2 = siamesenet(Variable(x0).cuda(), Variable(curr_aera).cuda())
                        euclidean_distance = F.pairwise_distance(output1, output2)
                        # print('Dissimilarity is %.2f\n ' % (euclidean_distance.item()))
                        # if euclidean_distance.item()<0.5:
                        #     filename1="temimg/v%d-f%d-t%d-siam%.2f-pre.JPEG"%(vidx,frame_idx,t,euclidean_distance.item())
                        #     # cv2.imwrite(filename1, x0.numpy())
                        #     cv2.imwrite(filename1, x0_crop)
                        #     filename2 = "temimg/v%d-f%d-t%d-siam%.2f-cur.JPEG"%(vidx,frame_idx,t,euclidean_distance.item())
                        #     cv2.imwrite(filename2, curr_aera_crop)
                        pre_aera[t_id] = curr_aera
                        pre_aera_crop[t_id] = curr_aera_crop

                        if euclidean_distance.item() > args.siam_thred:
                            #redect:
                            is_negative = True
                            dis_redet = 0
                            ts1 = time.time()
                            boxes, classes, scores = pred(predictor, class_names, frame)
                            ts2 = time.time()
                            spend_time['predict'] += ts2 - ts1
                            spend_time['n_predict_frames'] += 1
                            # frame_pred['frame_id'] = frame_idx
                            frame_pred['track_id'] = []
                            frame_pred['obj_name'] = []
                            frame_pred['bbox'] = []
                            frame_pred['score_cls'] = []
                            pre_aera = []
                            pre_aera_crop = []
                            ts1 = time.time()
                            n_bbox = len(boxes)
                            siam_thred_inf = []
                            for i_d in range(n_bbox):
                                t_aera, t_aera_crop, _ = transform3(frame, boxes[i_d])
                                pre_aera.append(t_aera)
                                pre_aera_crop.append(t_aera_crop)
                                if args.checktrackid:
                                    # calculate the id of the box with the highest similarity
                                    # if thred of the highest similarity is higher than 0.9,
                                    # n_trackid +1
                                    maxid = 0
                                    maxdistance = 9999
                                    for nt in range(len(obj_area)):
                                        output1, output2 = siamesenet(Variable(t_aera).cuda(),
                                                                      Variable(obj_area[nt]).cuda())
                                        euclidean_distance = F.pairwise_distance(output1, output2)
                                        if euclidean_distance.item() < maxdistance:
                                            maxdistance = euclidean_distance.item()
                                            maxid = nt
                                    # print("maxdistance: %.2f, iou: %.2f."%(maxdistance,cal_iou(boxes[i_d], obj_box[maxid])))
                                    siam_thred_inf.append(maxdistance)
                                    if maxdistance > args.siam_thred and cal_iou(boxes[i_d], obj_box[maxid]) < 0.6:
                                        n_trackid += 1
                                        obj_area.append(t_aera)
                                        obj_box.append(boxes[i_d])
                                        frame_pred['track_id'].append(n_trackid)
                                    else:
                                        frame_pred['track_id'].append(maxid)
                                        obj_box[maxid] = boxes[i_d]
                                else:
                                    # n_trackid_t=i_d
                                    frame_pred['track_id'].append(i_d)
                                    siam_thred_inf.append(0)
                                # frame_pred['track_id'].append(i_d)
                                frame_pred['obj_name'].append(classes[i_d])
                                frame_pred['bbox'].append(boxes[i_d])
                                frame_pred['score_cls'].append(scores[i_d])

                            ts2 = time.time()
                            spend_time['append'] += ts2 - ts1
                            break
                        else:
                            siam_thred_inf.append(euclidean_distance.item())
                            if args.checktrackid:
                                obj_box[frame_pred['track_id'][t_id]] = curr_bbox
            n_bbox=len(frame_pred['bbox'])
            if is_negative==False:
                spend_time['track'] += ts_all
                spend_time['n_track_frames'] += 1
                vid_pred['detortrack'].extend(np.full(n_bbox, 1))
                # if args.display_images:
                #     im_with_bb = display_result(frame, frame_pred['bbox'])  # draw box and display
                # else:
                #     im_with_bb = draw_boxes(frame, frame_pred['bbox'])
            else:
                vid_pred['detortrack'].extend(np.full(n_bbox, 0))
            ts1=time.time()
            vid_pred['frame_id'].extend(np.full(n_bbox, frame_pred['frame_id']))
            vid_pred['track_id'].extend(frame_pred['track_id'])
            vid_pred['obj_name'].extend(frame_pred['obj_name'])
            vid_pred['bbox'].extend(frame_pred['bbox'])
            vid_pred['score_cls'].extend(frame_pred['score_cls'])
            ts2=time.time()
            spend_time['append'] += ts2 - ts1
            spend_time['n_append'] += 1

        if args.display_images:
            cv2.resizeWindow("frame: %d"%frame_idx, 800, 800)
            if len(frame_pred['bbox']) == 0:
                cv2.imshow("result",frame)
                cv2.waitKey(1)
            else:
                boxes=np.asarray(frame_pred['bbox'])
                boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
                outputs={
                    "pred_boxes":boxes,
                    "scores":frame_pred['score_cls'],
                    "trackids":frame_pred['track_id'],
                    "pred_classes":frame_pred['obj_name'],
                    "detortrack":[vid_pred['detortrack'][-1]]*len(frame_pred['obj_name']),
                    "siam_inf":siam_thred_inf
                }
                v = Visualizer(frame[:, :, ::-1], metalog, scale=1.2)
                v = v.draw_instance_predictions2(outputs,args)
                cv2.imshow("result",v.get_image())
                cv2.waitKey(1)

        if args.save_result_images_bool:
            filename = os.path.join(args.save_result_images, str(frame_idx).rjust(4,'0')+'-99-21-final' + '.jpg')
            # cv2.imwrite(filename, im_with_bb)
            if len(frame_pred['bbox']) == 0:
                cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            else:
                boxes=np.asarray(frame_pred['bbox'])
                boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
                outputs={
                    "pred_boxes":boxes,
                    "scores":frame_pred['score_cls'],
                    "trackids": frame_pred['track_id'],
                    "pred_classes":frame_pred['obj_name'],
                    "detortrack":[vid_pred['detortrack'][-1]]*len(frame_pred['obj_name']),
                    "siam_inf":siam_thred_inf
                }
                v = Visualizer(frame[:, :, ::-1], metalog, scale=1.2)
                v = v.draw_instance_predictions2(outputs,args)
                cv2.imwrite(filename, v.get_image(), [int(cv2.IMWRITE_JPEG_QUALITY), 70])

        # record the curr_bbox result
        # bboxes[frame_idx] = curr_bbox

        '''
        # create or update storage + set iteration_range for training
        if frame_idx == 0:
            dataset_storage_pos = OnlineAdaptationDatasetStorage(initial_frame=frame, first_box=curr_bbox, opts=opts, args=args, positive=True)
            if opts['nNeg_init'] != 0:  # (thanks to small hack in adnet_test) the nNeg_online is also 0
                dataset_storage_neg = OnlineAdaptationDatasetStorage(initial_frame=frame, first_box=curr_bbox, opts=opts, args=args, positive=False)

            iteration_range = range(opts['finetune_iters'])
        else:
            assert dataset_storage_pos is not None
            if opts['nNeg_init'] != 0:  # (thanks to small hack in adnet_test) the nNeg_online is also 0
                assert dataset_storage_neg is not None

            # if confident or when always generate samples, generate new samples
            if ntraining < args.believe_score_result:
                always_generate_samples = True  # as FC7 wasn't trained, it is better to wait for some time to believe its confidence result to decide whether to generate samples or not.. Before believe it, better to just generate sample always
            else:
                always_generate_samples = False

            if always_generate_samples or (not is_negative or target_score > opts['successThre']):
                dataset_storage_pos.add_frame_then_generate_samples(frame, curr_bbox)

            iteration_range = range(opts['finetune_iters_online'])

        # training when depend on the frequency.. else, don't run the training code...
        if frame_idx % args.online_adaptation_every_I_frames == 0:
            ntraining += 1
            # generate dataset just before training
            dataset_pos = OnlineAdaptationDataset(dataset_storage_pos)
            data_loader_pos = data.DataLoader(dataset_pos, opts['minibatch_size'], num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True)
            batch_iterator_pos = None

            if opts['nNeg_init'] != 0:  # (thanks to small hack in adnet_test) the nNeg_online is also 0
                dataset_neg = OnlineAdaptationDataset(dataset_storage_neg)
                data_loader_neg = data.DataLoader(dataset_neg, opts['minibatch_size'], num_workers=args.num_workers,
                                                  shuffle=True, pin_memory=True)
                batch_iterator_neg = None
            else:
                dataset_neg = []

            epoch_size_pos = len(dataset_pos) // opts['minibatch_size']
            epoch_size_neg = len(dataset_neg) // opts['minibatch_size']
            epoch_size = epoch_size_pos + epoch_size_neg  # 1 epoch, how many iterations

            which_dataset = list(np.full(epoch_size_pos, fill_value=1))
            which_dataset.extend(np.zeros(epoch_size_neg, dtype=int))
            shuffle(which_dataset)

            print("1 epoch = " + str(epoch_size) + " iterations")

            if args.cuda:
                net.module.set_phase('train')
            else:
                net.set_phase('train')

            # training loop
            for iteration in iteration_range:
                all_iteration += 1  # use this for update the visualization
                #if all_iteration == 185:
                #    print("stop")
                # create batch iterator
                if (not batch_iterator_pos) or (iteration % epoch_size == 0):
                    batch_iterator_pos = iter(data_loader_pos)

                if opts['nNeg_init'] != 0:
                    if (not batch_iterator_neg) or (iteration % epoch_size == 0):
                        batch_iterator_neg = iter(data_loader_neg)

                # load train data
                if which_dataset[iteration % len(which_dataset)]:  # if positive
                    images, bbox, action_label, score_label = next(batch_iterator_pos)
                else:
                    images, bbox, action_label, score_label = next(batch_iterator_neg)

                if args.cuda:
                    images = torch.Tensor(images.cuda())
                    bbox = torch.Tensor(bbox.cuda())
                    action_label = torch.Tensor(action_label.cuda())
                    score_label = torch.Tensor(score_label.float().cuda())

                else:
                    images = torch.Tensor(images)
                    bbox = torch.Tensor(bbox)
                    action_label = torch.Tensor(action_label)
                    score_label = torch.Tensor(score_label)

                # forward
                t0 = time.time()
                action_out, score_out = net(images)

                # backprop
                optimizer.zero_grad()
                if which_dataset[iteration % len(which_dataset)]:  # if positive
                    action_l = action_criterion(action_out, torch.max(action_label, 1)[1])
                else:
                    action_l = torch.Tensor([0])
                score_l = score_criterion(score_out, score_label.long())
                loss = action_l + score_l
                loss.backward()
                optimizer.step()
                t1 = time.time()

                if all_iteration % 10 == 0:
                    print('iter ' + repr(all_iteration) + ' || Loss: %.4f ||' % (loss.data.item()), end=' ')
                    print('Timer: %.4f sec.' % (t1 - t0))
                    if args.visualize and args.send_images_to_visualization:
                        random_batch_index = np.random.randint(images.size(0))
                        writer.add_image('image', images.data[random_batch_index].cpu().numpy(), random_batch_index)

                if args.visualize:
                    writer.add_scalars('data/iter_loss', {'action_loss': action_l.item(),
                                                          'score_loss': score_l.item(),
                                                          'total': (action_l.item() + score_l.item())},
                                       global_step=all_iteration)
        '''
        t1_wholetracking = time.time()
        t_sum += t1_wholetracking - t0_wholetracking
        # print('whole tracking time = %.4f sec.' % (t1_wholetracking - t0_wholetracking))

    # evaluate the precision
    # bboxes = np.array(bboxes)
    # vid_info['gt'] = np.array(vid_info['gt'])

    # iou_precisions = iou_precision_plot(bboxes, vid_info['gt'], vid_path, show=args.display_images, save_plot=args.save_result_images)
    #
    # distance_precisions = distance_precision_plot(bboxes, vid_info['gt'], vid_path, show=args.display_images, save_plot=args.save_result_images)
    #
    # precisions = [distance_precisions, iou_precisions]

    # np.save(args.save_result_npy + '-bboxes.npy', bboxes)
    # np.save(args.save_result_npy + '-ground_truth.npy', vid_info['gt'])

    # return bboxes, t_sum, precisions
    print('vid %d : %d frames, whole tracking time : %.4f sec.' % (vidx,vid_info['nframes'],t_sum))
    if spend_time['n_predict_frames']!=0:
        print("predict time: %.2fs, predict frames: %d, average time: %.2fms."%(
            spend_time['predict'],spend_time['n_predict_frames'],
            (spend_time['predict']/spend_time['n_predict_frames'])*1000))
    if spend_time['n_track_frames']!=0:
        print("track time: %.2fs, track frames: %d, average time: %.2fms." % (
            spend_time['track'], spend_time['n_track_frames'],
            (spend_time['track'] / spend_time['n_track_frames']) * 1000))
    # if spend_time['n_readframe']!=0:
    #     print("readframe time: %.2fs, readframes: %d, average time: %.2fms." % (
    #         spend_time['readframe'], spend_time['n_readframe'],
    #         (spend_time['readframe'] / spend_time['n_readframe']) * 1000))
    # if spend_time['n_append']!=0:
    #     print("append time: %.2fs, n_append: %d, average time: %.2fms." % (
    #         spend_time['append'], spend_time['n_append'],
    #         (spend_time['append'] / spend_time['n_append']) * 1000))
    # if spend_time['n_transform']!=0:
    #     print("transform time: %.2fs, n_transform: %d, average time: %.2fms." % (
    #         spend_time['transform'], spend_time['n_transform'],
    #         (spend_time['transform'] / spend_time['n_transform']) * 1000))
    # if spend_time['n_cuda'] != 0:
    #     print(".cuda time: %.2fs, n_transform call: %d, average time: %.2fms." % (
    #         spend_time['cuda'], spend_time['n_cuda'],
    #         (spend_time['cuda'] / spend_time['n_cuda']) * 1000))

    # if spend_time['n_argmax_after_forward']!=0:
    #     print("argmax_after_forward time: %.2fs, n_argmax_after_forward: %d, average time: %.2fms." % (
    #         spend_time['argmax_after_forward'], spend_time['n_argmax_after_forward'],
    #         (spend_time['argmax_after_forward'] / spend_time['n_argmax_after_forward']) * 1000))
    # if spend_time['n_do_action']!=0:
    #     print("do_action time: %.2fs, n_do_action: %d, average time: %.2fms." % (
    #         spend_time['do_action'], spend_time['n_do_action'],
    #         (spend_time['do_action'] / spend_time['n_do_action']) * 1000))
    print('\n')
    return vid_pred,spend_time