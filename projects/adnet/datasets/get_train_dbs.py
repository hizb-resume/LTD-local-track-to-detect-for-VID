# matlab code:
# https://github.com/hellbell/ADNet/blob/3a7955587b5d395401ebc94a5ab067759340680d/train/get_train_dbs.m
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import numpy as np
import numpy.matlib

from utils.gen_samples import gen_samples
from utils.overlap_ratio import overlap_ratio
from utils.gen_action_labels import gen_action_labels


def get_train_dbs(vid_info, opts):
    img = cv2.imread(vid_info['img_files'][0])

    opts['scale_factor'] = 1.05
    opts['imgSize'] = list(img.shape)
    gt_skip = opts['train']['gt_skip']

    if vid_info['db_name'] == 'alov300':
        train_sequences = vid_info['gt_use'] == 1
    else:
        train_sequences = list(range(0, vid_info['nframes'], gt_skip))

    train_db_pos = []
    train_db_neg = []

    for train_i in range(len(train_sequences)):
        train_db_pos_ = {
            'img_path': [],
            'bboxes': [],
            'labels': [],
            'score_labels': []
        }
        train_db_neg_ = {
            'img_path': [],
            'bboxes': [],
            'labels': [],
            'score_labels': []
        }

        img_idx = train_sequences[train_i]
        gt_bbox = vid_info['gt'][img_idx]

        if len(gt_bbox) == 0:
            continue

        pos_examples = []
        while len(pos_examples) < opts['nPos_train']:
            pos = gen_samples('gaussian', gt_bbox, opts['nPos_train']*5, opts, 0.1, 5)
            r = overlap_ratio(pos, np.matlib.repmat(gt_bbox, len(pos), 1))
            pos = pos[np.array(r) > opts['posThre_train']]
            if len(pos) == 0:
                continue
            pos = pos[np.random.randint(low=0, high=len(pos),
                                        size=min(len(pos), opts['nPos_train']-len(pos_examples))), :]
            pos_examples.extend(pos)

        neg_examples = []
        while len(neg_examples) < opts['nNeg_train']:
            # in original code, this 1 line below use opts['nPos_train'] instead of opts['nNeg_train']
            neg = gen_samples('gaussian', gt_bbox, opts['nNeg_train']*5, opts, 2, 10)
            r = overlap_ratio(neg, np.matlib.repmat(gt_bbox, len(neg), 1))
            neg = neg[np.array(r) < opts['negThre_train']]
            if len(neg) == 0:
                continue
            neg = neg[np.random.randint(low=0, high=len(neg),
                                        size=min(len(neg), opts['nNeg_train']-len(neg_examples))), :]
            neg_examples.extend(neg)

        # examples = pos_examples + neg_examples
        action_labels_pos = gen_action_labels(opts['num_actions'], opts, np.array(pos_examples), gt_bbox)
        action_labels_neg = np.full((opts['num_actions'], len(neg_examples)), fill_value=-1)

        action_labels_pos = np.transpose(action_labels_pos).tolist()
        action_labels_neg = np.transpose(action_labels_neg).tolist()

        # action_labels = action_labels_pos + action_labels_neg

        train_db_pos_['img_path'] = np.full(len(pos_examples), vid_info['img_files'][img_idx])
        train_db_pos_['bboxes'] = pos_examples
        train_db_pos_['labels'] = action_labels_pos
        # score labels: 1 is positive. 0 is negative
        train_db_pos_['score_labels'] = list(np.ones(len(pos_examples), dtype=int))

        train_db_neg_['img_path'] = np.full(len(neg_examples), vid_info['img_files'][img_idx])
        train_db_neg_['bboxes'] = neg_examples
        train_db_neg_['labels'] = action_labels_neg
        # score labels: 1 is positive. 0 is negative
        train_db_neg_['score_labels'] = list(np.zeros(len(neg_examples), dtype=int))

        train_db_pos.append(train_db_pos_)
        train_db_neg.append(train_db_neg_)

    return train_db_pos, train_db_neg


# test the module
# from utils.get_train_videos import get_train_videos
# from utils.init_params import opts
# from utils.get_video_infos import get_video_infos
# train_videos = get_train_videos(opts)
# bench_name = train_videos['bench_names'][0]
# video_name = train_videos['video_names'][0]
# video_path = train_videos['video_paths'][0]
# vid_info = get_video_infos(bench_name, video_path, video_name)
# get_train_dbs(vid_info, opts)