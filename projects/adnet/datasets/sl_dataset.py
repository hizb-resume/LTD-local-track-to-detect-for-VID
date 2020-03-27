# pytorch dataset for SL learning
# matlab code (line 26-33):
# https://github.com/hellbell/ADNet/blob/master/train/adnet_train_SL.m
# reference:
# https://github.com/amdegroot/ssd.pytorch/blob/master/data/voc0712.py

import os,time
import cv2
import numpy as np
import torch
import torch.utils.data as data
from datasets.get_train_dbs import get_train_dbs
from datasets.get_train_dbs import get_train_dbs_ILSVR
from datasets.get_train_dbs import get_train_dbs_ILSVR_consecutive_frame
from utils.get_video_infos import get_video_infos


class SLDataset(data.Dataset):
    # train_videos = get_train_videos(opts)
    # train_videos = {  # the format of train_videos
    #         'video_names' : video_names,
    #         'video_paths' : video_paths,
    #         'bench_names' : bench_names
    #     }
    def __init__(self, train_db, transform=None):
        self.transform = transform
        self.train_db = train_db

    def __getitem__(self, index):
        image = cv2.imread(self.train_db['img_path'][index])
        frame2 = image.copy()
        frame2 = frame2.astype(np.float32)
        frame2 = torch.from_numpy(frame2).cuda()
        bboxes = self.train_db['bboxes'][index]
        action_labels = np.array(self.train_db['labels'][index], dtype=np.float32)
        score_labels = np.array(self.train_db['score_labels'][index])
        vid_idxs = self.train_db['vid_idx'][index]


        action_labels = torch.from_numpy(action_labels).cuda()
        score_labels = torch.from_numpy(score_labels.astype(np.float32)).cuda()
        vid_idxs = torch.Tensor(vid_idxs).cuda()
        if self.transform is not None:
            for i,bbox in enumerate(bboxes):
                # ims=None
                if i==0:
                # im, bbox, action_label, score_label = self.transform(frame2, bbox, action_labels[i], score_labels[i])
                    ims, _, _, _ = self.transform(frame2, bbox, action_labels[i], score_labels[i])
                else:
                    im, _, _, _ = self.transform(frame2, bbox, action_labels[i], score_labels[i])
                    ims=torch.cat([ims,im],dim=0)
        # return im, bbox, action_label, score_label, vid_idx
        bboxes = torch.Tensor(bboxes).cuda()
        return ims, bboxes, action_labels, score_labels, vid_idxs

    def __len__(self):
        return len(self.train_db['img_path'])

    #########################################################
    # ADDITIONAL FUNCTIONS

    def pull_image(self, index):
        im = cv2.imread(self.train_db['img_path'][index])
        return im

    def pull_anno(self, index):
        action_label = self.train_db['labels'][index]
        score_label = self.train_db['score_labels'][index]
        return action_label, score_label


def initialize_pos_neg_dataset(train_videos, opts,args, transform=None, multidomain=True):
    """
    Return list of pos and list of neg dataset for each domain.
    Args:
        train_videos:
        opts:
        transform:
        multidomain:
    Returns:
        datasets_pos: (list of SLDataset) List length: if multidomain, #videos (or domain). Else: 1
        datasets_neg: (list of SLDataset) List length: if multidomain, #videos (or domain). Else: 1
    """

    # datasets_pos = []
    # datasets_neg = []
    datasets_pos_neg = []

    if train_videos==None:
        num_videos=1
    else:
        num_videos = len(train_videos['video_names'])
    t0 = time.time()
    for vid_idx in range(num_videos):
        train_db = {
            'img_path': [],  # list of string
            'bboxes': [],  # list of ndarray left top coordinate [left top width height]
            'labels': [],  # list of ndarray #action elements. One hot vector
            'score_labels': [],  # list of scalar 0 (negative) or 1 (positive)
            'vid_idx': []  # list of int. Each video (or domain) index
        }
        # train_db_neg = {
        #     'img_path': [],  # list of string
        #     'bboxes': [],  # list of ndarray left top coordinate [left top width height]
        #     'labels': [],  # list of ndarray #action elements. One hot vector
        #     'score_labels': [],  # list of scalar 0 (negative) or 1 (positive)
        #     'vid_idx': []  # list of int. Each video (or domain) index
        # }

        if train_videos == None:
            print("generating dataset from ILSVR dataset...")
            # train_db_pos_, train_db_neg_ = get_train_dbs_ILSVR(opts)
            if args.train_consecutive:
                train_db_pos_neg_ = get_train_dbs_ILSVR_consecutive_frame(opts)
            else:
                train_db_pos_neg_ = get_train_dbs_ILSVR(opts)
        else:
            # print("generating dataset from video " + str(vid_idx + 1) + "/" + str(num_videos) +
            #   "(current total data (pos-neg): " + str(len(train_db_pos['labels'])) +
            #   "-" + str(len(train_db_neg['labels'])) + ")")
            print("generating dataset from video " + str(vid_idx + 1) + "/" + str(num_videos) +
                  "(current total data (pos+neg): " + str(len(train_db['labels']))  + ")")

            bench_name = train_videos['bench_names'][vid_idx]
            video_name = train_videos['video_names'][vid_idx]
            video_path = train_videos['video_paths'][vid_idx]
            vid_info = get_video_infos(bench_name, video_path, video_name)
            train_db_pos_, train_db_neg_ = get_train_dbs(vid_info, opts)
        # separate for each bboxes sample
        print("before train_db_pos['img_path'].extend", end=' : ')
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for sample_idx in range(len(train_db_pos_neg_)):
            # for img_path_idx in range(len(train_db_pos_[sample_idx]['score_labels'])):
            train_db['img_path'].append(train_db_pos_neg_[sample_idx]['img_path'])
            train_db['bboxes'].append(train_db_pos_neg_[sample_idx]['bboxes'])
            train_db['labels'].append(train_db_pos_neg_[sample_idx]['labels'])
            train_db['score_labels'].append(train_db_pos_neg_[sample_idx]['score_labels'])
            # train_db['vid_idx'].extend(np.repeat(vid_idx, len(train_db_pos_[sample_idx]['img_path'])))
            train_db['vid_idx'].append(vid_idx)

        #     if len(train_db_pos_neg_[sample_idx]['bboxes'])!=20:
        #         print("len(train_db_pos_neg_[sample_idx]['bboxes']): %d, img path: %s"%(
        #             len(train_db_pos_neg_[sample_idx]['bboxes']),train_db_pos_neg_[sample_idx]['img_path']))
        #     if len(train_db_pos_neg_[sample_idx]['labels'])!=20:
        #         print("len(train_db_pos_neg_[sample_idx]['labels']): %d, img path: %s"%(
        #             len(train_db_pos_neg_[sample_idx]['labels']),train_db_pos_neg_[sample_idx]['img_path']))
        #     if len(train_db_pos_neg_[sample_idx]['score_labels'])!=20:
        #         print("len(train_db_pos_neg_[sample_idx]['score_labels']): %d, img path: %s"%(
        #             len(train_db_pos_neg_[sample_idx]['score_labels']),train_db_pos_neg_[sample_idx]['img_path']))
        # print('over debug.')
        # print("\nFinish generating positive dataset... (current total data: " + str(len(train_db_pos['labels'])) + ")")

        # for sample_idx in range(len(train_db_neg_)):
        #     # for img_path_idx in range(len(train_db_neg_[sample_idx]['score_labels'])):
        #     train_db['img_path'].append(train_db_neg_[sample_idx]['img_path'])
        #     train_db['bboxes'].append(train_db_neg_[sample_idx]['bboxes'])
        #     train_db['labels'].append(train_db_neg_[sample_idx]['labels'])
        #     train_db['score_labels'].append(train_db_neg_[sample_idx]['score_labels'])
        #     # train_db['vid_idx'].extend(np.repeat(vid_idx, len(train_db_neg_[sample_idx]['img_path'])))
        #     train_db['vid_idx'].append(vid_idx)
        # print("\nFinish generating negative dataset... (current total data: " + str(len(train_db_neg['labels'])) + ")")

        print("after train_db_neg['img_path'].extend", end=' : ')
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        #dataset_pos = SLDataset(train_db_pos, transform=transform)
        dataset_pos_neg = SLDataset(train_db, transform=transform)
        print("after dataset_pos_neg = SLDataset(train_db", end=' : ')
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # dataset_neg = SLDataset(train_db_neg, transform=transform)
        # print("after dataset_neg = SLDataset(train_db_neg", end=' : ')
        # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        if multidomain:
            datasets_pos_neg.append(dataset_pos_neg)
            #datasets_neg.append(dataset_neg)
        else:
            if len(datasets_pos_neg)==0:
                datasets_pos_neg.append(dataset_pos_neg)
                #datasets_neg.append(dataset_neg)
                print("after datasets_pos_neg.append(dataset_pos_neg)", end=' : ')
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            else:
                # datasets_pos[0].train_db['img_path'].extend(dataset_pos.train_db['img_path'])
                # datasets_pos[0].train_db['bboxes'].extend(dataset_pos.train_db['bboxes'])
                # datasets_pos[0].train_db['labels'].extend(dataset_pos.train_db['labels'])
                # datasets_pos[0].train_db['score_labels'].extend(dataset_pos.train_db['score_labels'])
                # datasets_pos[0].train_db['vid_idx'].extend(dataset_pos.train_db['vid_idx'])
                #
                # datasets_neg[0].train_db['img_path'].extend(dataset_neg.train_db['img_path'])
                # datasets_neg[0].train_db['bboxes'].extend(dataset_neg.train_db['bboxes'])
                # datasets_neg[0].train_db['labels'].extend(dataset_neg.train_db['labels'])
                # datasets_neg[0].train_db['score_labels'].extend(dataset_neg.train_db['score_labels'])
                # datasets_neg[0].train_db['vid_idx'].extend(dataset_neg.train_db['vid_idx'])
                datasets_pos_neg[0].train_db['img_path'].extend(dataset_pos_neg.train_db['img_path'])
                datasets_pos_neg[0].train_db['bboxes'].extend(dataset_pos_neg.train_db['bboxes'])
                datasets_pos_neg[0].train_db['labels'].extend(dataset_pos_neg.train_db['labels'])
                datasets_pos_neg[0].train_db['score_labels'].extend(dataset_pos_neg.train_db['score_labels'])
                datasets_pos_neg[0].train_db['vid_idx'].extend(dataset_pos_neg.train_db['vid_idx'])

    t1 = time.time()
    all_time = t1 - t0
    all_m = all_time // 60
    all_s = all_time % 60
    print('time of generating dataset: %d m  %d s (%d s)' % (all_m, all_s, all_time))
    # return datasets_pos, datasets_neg
    return datasets_pos_neg

