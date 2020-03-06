import argparse
import sys
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
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

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='ADNet eval')
parser.add_argument('--weight_file', default='weights/ADNet_SL_epoch14_final.pth', type=str, help='The pretrained weight file')
parser.add_argument('--weight_detector', default='../datasets/tem/train_output/model_0599999.pth', type=str, help='The pretrained weight file of detector')
parser.add_argument('--weight_siamese', default='siameseWeight2/SiameseNet_epoch19_final.pth', type=str, help='The pretrained weight file of siamesenet')
parser.add_argument('--results_file', default='../datasets/data/ILSVRC-vid-eval-tem', type=str, help='The eval results file')
parser.add_argument('--v_start_id', default=0, type=int, help='The start no of eval videos')
parser.add_argument('--v_end_id', default=0, type=int, help='The end no of eval videos')
parser.add_argument('--track', default=True, type=str2bool, help='track between detect')
parser.add_argument('--siam_thred', default=0.9, type=float, help='similarity thred between frames')
parser.add_argument('--eval_imgs', default=0, type=int, help='the num of imgs that picked from val.txt, 0 represent all imgs')
parser.add_argument('--gt_skip', default=5, type=int, help='frame sampling frequency')
parser.add_argument('--test1vid', default=False, type=str2bool, help='only test 1 video')

parser.add_argument('--num_workers', default=6, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--visualize', default=False, type=str2bool, help='Use tensorboardx to for visualization')
# parser.add_argument('--send_images_to_visualization', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--display_images', default=False, type=str2bool, help='Whether to display images or not')
parser.add_argument('--save_result_images_bool', default=False, type=str2bool, help='save results folder')
parser.add_argument('--save_result_images', default='save_result_images2', type=str, help='save results folder')
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


def testsiamese(siamesenet,videos_infos):
    transform3_adition = transforms.Compose([transforms.Resize((100, 100)),
                                             transforms.ToTensor()
                                             ])
    transform3 = ADNet_Augmentation3(transform3_adition)
    vlen=len(videos_infos)
    for i in range(500):
        vidx1 = random.randint(0, vlen-1)
        fidx1=random.randint(0,videos_infos[vidx1]['nframes']-1)
        frame_path1 = videos_infos[vidx1]['img_files'][fidx1]
        frame1 = cv2.imread(frame_path1)
        gt1=videos_infos[vidx1]['gt'][fidx1][0]
        t_aera1, _, _ = transform3(frame1, gt1)

        while True:
            vidx2 = random.randint(0, vlen-1)
            if vidx2 != vidx1:
                break
        fidx2 = random.randint(0, videos_infos[vidx2]['nframes']-1)
        frame_path2 = videos_infos[vidx2]['img_files'][fidx2]
        frame2 = cv2.imread(frame_path2)
        gt2 = videos_infos[vidx2]['gt'][fidx2][0]
        t_aera2, _, _ = transform3(frame2, gt2)

        output1, output2 = siamesenet(Variable(t_aera1).cuda(), Variable(t_aera2).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        print(' \n%.2f ' % (euclidean_distance.item()),end='  ')
        if euclidean_distance.item()<0.8:
            print("vid1: %d, name1: %s; vid2: %d , name2: %s."%(
                vidx1,videos_infos[vidx1]['name'][fidx1][0],vidx2,videos_infos[vidx2]['name'][fidx2][0]),end='  ')


if __name__ == "__main__":
    args = parser.parse_args()

    # cfg = get_cfg()
    # cfg.merge_from_file("../../../configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
    # cfg.MODEL.WEIGHTS = "../../../demo/faster_rcnn_R_101_FPN_3x.pkl"
    # metalog=MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

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

    siamesenet = SiameseNetwork().cuda()
    resume = args.weight_siamese
    # resume = False
    if resume:
        siamesenet.load_weights(resume)
        checkpoint = torch.load(resume)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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

    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print(
                "WARNING: It looks like you have a CUDA device, but aren't " + "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')



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
        vid_path = '../datasets/data/ILSVRC/Data/VID/train/ILSVRC2015_VID_train_0001/ILSVRC2015_train_00165005/'
        vid_folder = vid_path.split('/')[-2]
        # vid_path = "../../../demo/examples/jiaotong2.avi"
        # vid_folder=vid_path.split('/')[-1]
        if args.save_result_images_bool:
            args.save_result_images = os.path.join(args.save_result_images, vid_folder)
            if not os.path.exists(args.save_result_images):
                os.makedirs(args.save_result_images)
        vid_pred = adnet_test(net, predictor, siamesenet, metalog, class_names, 0, vid_path, opts, args)
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

        t_eval0=time.time()
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

            vid_pred = adnet_test(net,predictor,siamesenet,metalog,class_names, vidx,vid_folder['img_files'], opts, args)
            gen_pred_file(args.results_file,vid_pred)
        #     all_precisions.append(precisions)
        #
        # print(all_precisions)
        t_eval1 = time.time()
        all_time = t_eval1 - t_eval0
        all_d = all_time // 86400
        all_h = all_time % 86400 // 3600
        all_m = all_time % 3600 // 60
        all_s = all_time % 60
        print("eval time cost: %d d %d h %d m %d s ."% (all_d,all_h,all_m,all_s))
    