import argparse
import sys
import os
import multiprocessing
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
# sys.path.append("..")
# import _init_paths

# dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, dir_mytest)

import _init_paths
from options.general2 import opts
from models.ADNet import adnet
from utils.get_train_videos import get_train_videos
from utils.ADNet_evalTools import gen_pred_file
from utils.my_util import get_ILSVRC_eval_infos
from utils.siamese_test_gui import siamese_test
from utils.gen_samples import gen_samples
from utils.display import draw_box,draw_box_bigline
from utils.augmentations import ADNet_Augmentation3
from trainers.adnet_test import adnet_test
from datasets.ILSVRC import register_ILSVRC
from models.SiameseNet import SiameseNetwork
import torch
torch.multiprocessing.set_start_method('spawn', force=True)
import torch.backends.cudnn as cudnn
import torch.nn as nn
import time
import cv2
import random
import glob
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# from PyQt5 import QtWidgets, QtCore, QtGui
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
# from PyQt5.QtCore import *
# from PyQt5.QtWidgets import QApplication, QWidget

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='ADNet eval')
parser.add_argument('--weight_file', default='weights2/ADNet_SL_epoch24_10000.pth', type=str, help='The pretrained weight file')
parser.add_argument('--weight_detector', default='../datasets/tem/train_output/model_0599999.pth', type=str, help='The pretrained weight file of detector')
parser.add_argument('--weight_siamese', default='siameseWeight2/SiameseNet_epoch19_final.pth', type=str, help='The pretrained weight file of siamesenet')
parser.add_argument('--results_file', default='../datasets/data/ILSVRC-vid-eval-delete', type=str, help='The eval results file')
parser.add_argument('--v_start_id', default=0, type=int, help='The start no of eval videos')
parser.add_argument('--v_end_id', default=0, type=int, help='The end no of eval videos')
parser.add_argument('--track', default=True, type=str2bool, help='track between detect')
parser.add_argument('--siam_thred', default=0.9, type=float, help='similarity thred between frames')
parser.add_argument('--update_siam_thred', default=0.4, type=float, help='update obj area when thred< the value')
parser.add_argument('--eval_imgs', default=1000, type=int, help='the num of imgs that picked from val.txt, 0 represent all imgs')
parser.add_argument('--gt_skip', default=5, type=int, help='frame sampling frequency')
parser.add_argument('--dataset_year', default=2015, type=int, help='dataset version, like ILSVRC2015, ILSVRC2017')
parser.add_argument('--useSiamese', default=True, type=str2bool, help='use siamese or not')
parser.add_argument('--checktrackid', default=False, type=str2bool, help='if objects in different frames are the same instance, trackid should be same too')

parser.add_argument('--multi_cpu_eval', default=False, type=str2bool, help='mutiple cpu or not when evaluate')
parser.add_argument('--testSiamese', default=False, type=str2bool, help='test siamese or not')
parser.add_argument('--test1vid', default=False, type=str2bool, help='only test 1 video')
parser.add_argument('--testVidPath', default='../datasets/data/ILSVRC/Data/VID/val/ILSVRC2015_val_00136000/',
                    type=str, help='test video path, only turn on when --test1vid is True')
parser.add_argument('--label_more', default=False, type=str2bool, help='show tack_det/siamese thred in labels or not')

parser.add_argument('--num_workers', default=6, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--visualize', default=False, type=str2bool, help='Use tensorboardx to for visualization')
# parser.add_argument('--send_images_to_visualization', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--display_images', default=False, type=str2bool, help='Whether to display images or not')
parser.add_argument('--save_result_images_bool', default=False, type=str2bool, help='save results folder')
parser.add_argument('--save_result_images', default='save_result_images', type=str, help='save results folder')
parser.add_argument('--display_images_t', default=False, type=str2bool, help='display t patches between frames')
parser.add_argument('--save_result_images_t', default=False, type=str2bool, help='save t patches between frames')
# parser.add_argument('--save_result_npy', default='results_on_test_images_part2', type=str, help='save results folder')

# parser.add_argument('--initial_samples', default=3000, type=int, help='Number of training samples for the first frame. N_I')
# parser.add_argument('--online_samples', default=250, type=int, help='Number of training samples for the other frames. N_O')
# parser.add_argument('--redetection_samples', default=256, type=int, help='Number of samples for redetection. N_det')
# parser.add_argument('--initial_iteration', default=300, type=int, help='Number of iteration in initial training. T_I')
# parser.add_argument('--online_iteration', default=30, type=int, help='Number of iteration in online training. T_O')
# parser.add_argument('--online_adaptation_every_I_frames', default=10, type=int, help='Frequency of online training. I')
# parser.add_argument('--number_past_frames', default=20, type=int, help='The training data were sampled from the past J frames. J')

parser.add_argument('--believe_score_result', default=0, type=int, help='Believe score result after n training')

# parser.add_argument('--pos_samples_ratio', default='0.5', type=float,
#                     help='The ratio of positive in all samples for online adaptation. Rest of it will be negative samples. Default: 0.5')


def process_adnet_test(videos_infos,dataset_start_id, v_start_id,v_end_id,train_videos,save_root,
                        spend_times_share,vid_preds,net, siamesenet, opts,args, lock):
    register_ILSVRC()
    cfg = get_cfg()
    cfg.merge_from_file("../../../configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
    # cfg.MODEL.WEIGHTS ="../datasets/tem/train_output/model_0449999.pth"
    cfg.MODEL.WEIGHTS = args.weight_detector
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 30
    metalog = MetadataCatalog.get("ILSVRC_VID_val")

    predictor = DefaultPredictor(cfg)
    class_names = metalog.get("thing_classes", None)
    for vidx in range(v_start_id, v_end_id):
        # for vidx in range(20):
        vid_folder = videos_infos[vidx]

        if args.save_result_images_bool:
            args.save_result_images = os.path.join(save_root, train_videos['video_names'][vidx])
            if not os.path.exists(args.save_result_images):
                os.makedirs(args.save_result_images)

        vid_pred,spend_time = adnet_test(net,predictor,siamesenet,metalog,class_names, vidx,vid_folder['img_files'], opts, args)
        try:
            lock.acquire()
            spend_times=spend_times_share[0].copy()
            spend_times['predict']+=spend_time['predict']
            spend_times['n_predict_frames'] += spend_time['n_predict_frames']
            spend_times['track'] += spend_time['track']
            spend_times['n_track_frames'] += spend_time['n_track_frames']
            spend_times['readframe'] += spend_time['readframe']
            spend_times['n_readframe'] += spend_time['n_readframe']
            spend_times['append'] += spend_time['append']
            spend_times['n_append'] += spend_time['n_append']
            spend_times['transform'] += spend_time['transform']
            spend_times['n_transform'] += spend_time['n_transform']
            spend_times['argmax_after_forward'] += spend_time['argmax_after_forward']
            spend_times['n_argmax_after_forward'] += spend_time['n_argmax_after_forward']
            spend_times['do_action'] += spend_time['do_action']
            spend_times['n_do_action'] += spend_time['n_do_action']
            spend_times_share[0]=spend_times
            vid_preds[vidx-dataset_start_id]=vid_pred
        except Exception as err:
            raise err
        finally:
            lock.release()

if __name__ == "__main__":
    args = parser.parse_args()

    # cfg = get_cfg()
    # cfg.merge_from_file("../../../configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
    # cfg.MODEL.WEIGHTS = "../../../demo/faster_rcnn_R_101_FPN_3x.pkl"
    # metalog=MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    
    siamesenet=''
    if args.useSiamese:
        siamesenet = SiameseNetwork().cuda()
        resume = args.weight_siamese
        # resume = False
        if resume:
            siamesenet.load_weights(resume)
        # checkpoint = torch.load(resume)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print(
                "WARNING: It looks like you have a CUDA device, but aren't " + "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if not args.testSiamese:
        if not args.multi_cpu_eval:
            register_ILSVRC()
            cfg = get_cfg()
            cfg.merge_from_file("../../../configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
            # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
            # cfg.MODEL.WEIGHTS ="../datasets/tem/train_output/model_0449999.pth"
            cfg.MODEL.WEIGHTS = args.weight_detector
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 30
            metalog = MetadataCatalog.get("ILSVRC_VID_val")

            predictor = DefaultPredictor(cfg)
            class_names = metalog.get("thing_classes", None)

    # assert 0 < args.pos_samples_ratio <= 1, "the pos_samples_ratio valid range is (0, 1]"

    # set opts based on the args.. especially the number of samples etc.
    # opts['nPos_init'] = int(args.initial_samples * args.pos_samples_ratio)
    # opts['nNeg_init'] = int(args.initial_samples - opts['nPos_init'])
    # opts['nPos_online'] = int(args.online_samples * args.pos_samples_ratio)
    # opts['nNeg_online'] = int(args.online_samples - opts['nPos_online'])

    # just to make sure if one of nNeg is zero, the other nNeg is zero (kinda small hack...)
    # if opts['nNeg_init'] == 0:
    #     opts['nNeg_online'] = 0
    #     opts['nPos_online'] = args.online_samples
    #
    # elif opts['nNeg_online'] == 0:
    #     opts['nNeg_init'] = 0
    #     opts['nPos_init'] = args.initial_samples

    # opts['finetune_iters'] = args.initial_iteration
    # opts['finetune_iters_online'] = args.online_iteration
    # opts['redet_samples'] = args.redetection_samples

        if args.save_result_images_bool:
            args.save_result_images = os.path.join(args.save_result_images,
                                                   os.path.basename(args.weight_file)[:-4])
            if not os.path.exists(args.save_result_images):
                os.makedirs(args.save_result_images)

    # args.save_result_npy = os.path.join(args.save_result_npy, os.path.basename(args.weight_file)[:-4] + '-' +
    #                                     str(args.pos_samples_ratio))
    # if not os.path.exists(args.save_result_npy):
    #     os.makedirs(args.save_result_npy)

    



    # dataset_root = os.path.join('../datasets/data', opts['test_db'])
    # vid_folders = []
    # for filename in os.listdir(dataset_root):
    #     if os.path.isdir(os.path.join(dataset_root,filename)):
    #         vid_folders.append(filename)
    # vid_folders.sort(key=str.lower)
    # all_precisions = []

        save_root = args.save_result_images
    # save_root_npy = args.save_result_npy

        print('Loading {}...'.format(args.weight_file))
        opts['num_videos'] = 1
        net, domain_nets = adnet(opts, trained_file=args.weight_file, random_initialize_domain_specific=False)
        net.eval()
        if args.cuda:
            net = nn.DataParallel(net)
            cudnn.benchmark = True
        if args.cuda:
            net = net.cuda()
        if args.cuda:
            net.module.set_phase('test')
        else:
            net.set_phase('test')

    if args.test1vid:
        vid_path = args.testVidPath
        vid_folder = vid_path.split('/')[-2]
        # vid_path = "../../../demo/examples/jiaotong2.avi"
        # vid_folder=vid_path.split('/')[-1]
        if args.save_result_images_bool:
            args.save_result_images = os.path.join(args.save_result_images, vid_folder)
            if not os.path.exists(args.save_result_images):
                os.makedirs(args.save_result_images)
        vid_pred = adnet_test(net, predictor, siamesenet, metalog, class_names, 0, vid_path, opts, args)
    elif args.testSiamese:
        videos_infos, train_videos = get_ILSVRC_eval_infos(args)
        transform3_adition = transforms.Compose([transforms.Resize((100, 100)),
                                                 transforms.ToTensor()
                                                 ])
        transform3 = ADNet_Augmentation3(transform3_adition)
        from PyQt5 import QtWidgets
        app = QtWidgets.QApplication(sys.argv)
        st = siamese_test(siamesenet,videos_infos,transform3)
        st.show()
        sys.exit(app.exec_())
    else:
        videos_infos, train_videos = get_ILSVRC_eval_infos(args)

        # testsiamese(siamesenet,videos_infos)

        v_start_id=args.v_start_id
        v_end_id=args.v_end_id
        if v_start_id<0:
            v_start_id=0
        elif v_start_id>=len(videos_infos):
            v_start_id=len(videos_infos)-1
        if v_end_id ==0 or v_end_id>len(videos_infos):
            v_end_id=len(videos_infos)

        print("videos nums: %d ." % (v_end_id-v_start_id))

        spend_times = {
            'predict': 0,
            'n_predict_frames': 0,
            'track': 0,
            'n_track_frames': 0,
            'readframe': 0,
            'n_readframe': 0,
            'append': 0,
            'n_append': 0,
            'transform': 0,
            'n_transform': 0,
            # 'cuda':0,
            # 'n_cuda':0,
            'argmax_after_forward': 0,
            'n_argmax_after_forward': 0,
            'do_action': 0,
            'n_do_action': 0
        }

        t_eval0=time.time()

        if args.multi_cpu_eval:
            all_vid_num=v_end_id-v_start_id
            cpu_num = 5
            if all_vid_num < cpu_num:
                cpu_num = all_vid_num
            every_cpu_vid = all_vid_num // cpu_num
            start_vid=[]
            end_vid=[]
            for gn in range(cpu_num - 1):
                # gn * every_gpu_img: (gn + 1) * every_gpu_img
                start_vid.append(gn * every_cpu_vid+v_start_id)
                end_vid.append((gn + 1) * every_cpu_vid+v_start_id)
            start_vid.append((cpu_num-1) * every_cpu_vid+v_start_id)
            end_vid.append(v_end_id)

            vid_preds = multiprocessing.Manager().list()
            spend_times_mul = multiprocessing.Manager().list()
            spend_times_mul.append(spend_times)
            for i in range(all_vid_num):
                vid_preds.append([])

            lock = multiprocessing.Manager().Lock()
            record = []
            for i in range(cpu_num):
                # print("fragment %d: start_vid: %d, end_vid: %d."%(i,start_vid[i],end_vid[i]))
                process = multiprocessing.Process(target=process_adnet_test,
                                                  args=(videos_infos, v_start_id,start_vid[i],end_vid[i],train_videos,save_root,
                                                        spend_times_mul,vid_preds,net, siamesenet, opts,args, lock))
                process.start()
                record.append(process)
            for process in record:
                process.join()

            vid_preds=list(vid_preds)
            spend_times=list(spend_times_mul)[0]
            for i in range(len(vid_preds)):
                isstart = i == 0
                gen_pred_file(args.results_file, vid_preds[i], isstart)
        else:
            # for vidx,vid_folder in enumerate(videos_infos):
            for vidx  in range(v_start_id,v_end_id):
            # for vidx in range(20):
                vid_folder=videos_infos[vidx]

                # net, domain_nets = adnet(opts, trained_file=args.weight_file, random_initialize_domain_specific=True)
                # net.train()
                if args.save_result_images_bool:
                    args.save_result_images = os.path.join(save_root, train_videos['video_names'][vidx])
                    if not os.path.exists(args.save_result_images):
                        os.makedirs(args.save_result_images)

                # args.save_result_npy = os.path.join(save_root_npy, train_videos['video_names'][vidx])

                # vid_path = os.path.join(train_videos['video_paths'][vidx], train_videos['video_names'][vidx])

                # load ADNetDomainSpecific

                # if args.cuda:
                #     net.module.load_domain_specific(domain_nets[0])
                # else:
                #     net.load_domain_specific(domain_nets[0])

                vid_pred,spend_time = adnet_test(net,predictor,siamesenet,metalog,class_names, vidx,vid_folder['img_files'], opts, args)
                isstart=vidx==v_start_id
                gen_pred_file(args.results_file,vid_pred,isstart)

                spend_times['predict']+=spend_time['predict']
                spend_times['n_predict_frames'] += spend_time['n_predict_frames']
                spend_times['track'] += spend_time['track']
                spend_times['n_track_frames'] += spend_time['n_track_frames']
                spend_times['readframe'] += spend_time['readframe']
                spend_times['n_readframe'] += spend_time['n_readframe']
                spend_times['append'] += spend_time['append']
                spend_times['n_append'] += spend_time['n_append']
                spend_times['transform'] += spend_time['transform']
                spend_times['n_transform'] += spend_time['n_transform']
                spend_times['argmax_after_forward'] += spend_time['argmax_after_forward']
                spend_times['n_argmax_after_forward'] += spend_time['n_argmax_after_forward']
                spend_times['do_action'] += spend_time['do_action']
                spend_times['n_do_action'] += spend_time['n_do_action']

        #     all_precisions.append(precisions)
        #
        # print(all_precisions)
        t_eval1 = time.time()
        all_time = t_eval1 - t_eval0
        all_d = all_time // 86400
        all_h = all_time % 86400 // 3600
        all_m = all_time % 3600 // 60
        all_s = all_time % 60
        print("all vids eval time cost: %d d %d h %d m %d s .\n"% (all_d,all_h,all_m,all_s))

        if spend_times['n_predict_frames'] != 0:
            print("whole predict time: %.2fs, predict frames: %d, average time: %.2fms." % (
                spend_times['predict'], spend_times['n_predict_frames'],
                (spend_times['predict'] / spend_times['n_predict_frames']) * 1000))
        if spend_times['n_track_frames'] != 0:
            print("whole track time: %.2fs, track frames: %d, average time: %.2fms." % (
                spend_times['track'], spend_times['n_track_frames'],
                (spend_times['track'] / spend_times['n_track_frames']) * 1000))
        if spend_times['n_readframe'] != 0:
            print("whole readframe time: %.2fs, readframes: %d, average time: %.2fms." % (
                spend_times['readframe'], spend_times['n_readframe'],
                (spend_times['readframe'] / spend_times['n_readframe']) * 1000))
        if spend_times['n_append'] != 0:
            print("whole append time: %.2fs, n_append: %d, average time: %.2fms." % (
                spend_times['append'], spend_times['n_append'],
                (spend_times['append'] / spend_times['n_append']) * 1000))
        if spend_times['n_transform'] != 0:
            print("whole transform time: %.2fs, n_transform: %d, average time: %.2fms." % (
                spend_times['transform'], spend_times['n_transform'],
                (spend_times['transform'] / spend_times['n_transform']) * 1000))
        # if spend_times['n_cuda'] != 0:
        #     print(".cuda time: %.2fs, n_transform call: %d, average time: %.2fms." % (
        #         spend_times['cuda'], spend_times['n_cuda'],
        #         (spend_times['cuda'] / spend_times['n_cuda']) * 1000))

        if spend_times['n_argmax_after_forward'] != 0:
            print("whole argmax_after_forward time: %.2fs, n_argmax_after_forward: %d, average time: %.2fms." % (
                spend_times['argmax_after_forward'], spend_times['n_argmax_after_forward'],
                (spend_times['argmax_after_forward'] / spend_times['n_argmax_after_forward']) * 1000))
        if spend_times['n_do_action'] != 0:
            print("whole do_action time: %.2fs, n_do_action: %d, average time: %.2fms." % (
                spend_times['do_action'], spend_times['n_do_action'],
                (spend_times['do_action'] / spend_times['n_do_action']) * 1000))
        print('\n')
    