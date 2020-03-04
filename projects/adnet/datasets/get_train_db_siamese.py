# matlab code:
# https://github.com/hellbell/ADNet/blob/3a7955587b5d395401ebc94a5ab067759340680d/train/get_train_dbs.m
import sys,os,time
import multiprocessing
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import numpy as np
import numpy.matlib

from utils.gen_samples import gen_samples
from utils.overlap_ratio import overlap_ratio
from utils.gen_action_labels import gen_action_labels
from utils.my_util import get_xml_box_label
from utils.my_util import get_xml_img_size
from utils.my_util import get_xml_img_info
from utils.my_util import get_siamese_train_infos

def process_data_siamese(img_paths, opt, train_db_pos_neg_all, lock):
    opts=opt.copy()
    train_db_pos_neg_gpu = []
    # train_db_neg_gpu = []
    for train_i in img_paths:
        train_db_pos_ = {
            'img_path': '',
            'bboxes': [],
            'labels': [],
            'score_labels': []
        }
        train_db_neg_ = {
            'img_path': '',
            'bboxes': [],
            'labels': [],
            'score_labels': []
        }


        #img_idx = train_sequences[train_i]
        #gt_bbox = vid_info['gt'][img_idx]

        #if len(gt_bbox) == 0:
        #    continue
        gt_file_path='../datasets/data/ILSVRC/Annotations/VID/train/'+train_i+'.xml'
        #gt_bbox=get_xml_box_label(gt_file_path)
        #opts['imgSize'] = get_xml_img_size(gt_file_path)
        imginfo=get_xml_img_info(gt_file_path)
        gt_bboxs=imginfo['gts']
        opts['imgSize'] =imginfo['imgsize']
        img_path = '../datasets/data/ILSVRC/Data/VID/train/' + train_i + '.JPEG'
        for gt_bbox in gt_bboxs:
            train_db_pos_neg = {
                'img_path': '',
                'bboxes': [],
                'labels': [],
                'score_labels': []
            }
            pos_examples = []
            while len(pos_examples) < opts['nPos_train']:
                pos = gen_samples('gaussian', gt_bbox, opts['nPos_train']*5, opts, 0.1, 5)
                r = overlap_ratio(pos, np.matlib.repmat(gt_bbox, len(pos), 1))
                pos = pos[np.array(r) > opts['posThre_train']]
                if len(pos) == 0:
                    #continue
                    break
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
                    #continue
                    break
                neg = neg[np.random.randint(low=0, high=len(neg),
                                            size=min(len(neg), opts['nNeg_train']-len(neg_examples))), :]
                neg_examples.extend(neg)

            # examples = pos_examples + neg_examples
            action_labels_pos = gen_action_labels(opts['num_actions'], opts, np.array(pos_examples), gt_bbox)
            action_labels_neg = np.full((opts['num_actions'], len(neg_examples)), fill_value=-1)

            action_labels_pos = np.transpose(action_labels_pos).tolist()
            action_labels_neg = np.transpose(action_labels_neg).tolist()

            # action_labels = action_labels_pos + action_labels_neg


            # train_db_pos_['bboxes'].extend(pos_examples)
            # train_db_pos_['labels'].extend(action_labels_pos)
            # # score labels: 1 is positive. 0 is negative
            # train_db_pos_['score_labels'].extend(list(np.ones(len(pos_examples), dtype=int)))
            #
            #
            # train_db_neg_['bboxes'].extend(neg_examples)
            # train_db_neg_['labels'].extend(action_labels_neg)
            # # score labels: 1 is positive. 0 is negative
            # train_db_neg_['score_labels'].extend(list(np.zeros(len(neg_examples), dtype=int)))

            train_db_pos_neg['bboxes'].extend(pos_examples)
            train_db_pos_neg['labels'].extend(action_labels_pos)
            # score labels: 1 is positive. 0 is negative
            train_db_pos_neg['score_labels'].extend(list(np.ones(len(pos_examples), dtype=int)))


            train_db_pos_neg['bboxes'].extend(neg_examples)
            train_db_pos_neg['labels'].extend(action_labels_neg)
            # score labels: 1 is positive. 0 is negative
            train_db_pos_neg['score_labels'].extend(list(np.zeros(len(neg_examples), dtype=int)))

            train_db_pos_neg['img_path'] = img_path
        # train_db_pos_['img_path'] = img_path
        # train_db_neg_['img_path'] = img_path

            # if len(train_db_pos_['bboxes']) != 0 and len(train_db_neg_['bboxes']) != 0:
            #     train_db_pos_gpu.append(train_db_pos_)
            #     train_db_neg_gpu.append(train_db_neg_)
            if len(train_db_pos_neg['bboxes']) == (opts['nPos_train']+ opts['nNeg_train']):
                train_db_pos_neg_gpu.append(train_db_pos_neg)
                # train_db_neg_gpu.append(train_db_neg_)
            # box_ii += 1

        # img_ii += 1

        # if img_ii==3471:
        #     print("when gt_skip set to 200, and the img_ii=3472, the gen_samples function can't produce examples that iou>thred")
        #     #'ILSVRC2015_VID_train_0002/ILSVRC2015_train_00633000/000025'
            #reason:the img is so small and unclear
        # if img_ii%1000==0 and img_ii!=0:
        #     t9=time.time()
        #     real_time=t9-t2
        #     all_time=t9-t0
        #     all_h=all_time//3600
        #     all_m=all_time%3600//60
        #     all_s=all_time%60
        #     speed_img=1000/real_time
        #     speed_box=(box_ii-box_ii_start)/real_time
        #     all_speed_img=img_ii/all_time
        #     all_speed_box = box_ii/all_time
        #     print('\ndone imgs: %d , done boxes: %d , all imgs: %d. '%(img_ii,box_ii,all_img_num))
        #     print('real_time speed: %d imgs/s, %d boxes/s'%(speed_img,speed_box))
        #     print('avg_time speed: %d imgs/s, %d boxes/s' % (all_speed_img, all_speed_box))
        #     print('spend time: %d h  %d m  %d s (%d s)'%(all_h,all_m,all_s,all_time))
        #     box_ii_start=box_ii
        #     t2=time.time()
    try:
        lock.acquire()
        # print("len(train_db_pos_gpu): %d" % len(train_db_pos_gpu))
        train_db_pos_neg_all.extend(train_db_pos_neg_gpu)
        # print("len(train_db_pos): %d" % len(train_db_pos))
        # print("len(train_db_neg_gpu): %d" % len(train_db_neg_gpu))
        # train_db_neg.extend(train_db_neg_gpu)
        # print("len(train_db_neg): %d" % len(train_db_neg))
    except Exception as err:
        raise err
    finally:
        lock.release()
    #lock.acquire()
    #print(sign, os.getpid())
    #lock.release()

def get_train_dbs_siamese(opts):
    # opts['scale_factor'] = 1.05
    # gt_skip = opts['train']['gt_skip']

    print('before get_train_dbs_ILSVR', end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    train_db_pos_neg = multiprocessing.Manager().list()

    videos_infos= get_siamese_train_infos()

    all_vid_num = len(videos_infos)

    # cpu_num=27
    gpu_num = 24
    if all_vid_num<gpu_num:
        gpu_num=all_vid_num
    every_gpu_img=all_vid_num//gpu_num
    vid_paths_as=[]
    for gn in range(gpu_num-1):
        vid_paths_as.append(videos_infos[gn*every_gpu_img:(gn+1)*every_gpu_img])
    vid_paths_as.append(videos_infos[(gpu_num-1) * every_gpu_img:])

    lock = multiprocessing.Manager().Lock()
    record = []
    for i in range(gpu_num):
        process = multiprocessing.Process(target=process_data_siamese, args=(vid_paths_as[i], opts,train_db_pos_neg,lock))
        process.start()
        record.append(process)
    for process in record:
        process.join()

    print('before train_db_pos=list(train_db_pos_neg)', end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train_db_pos_neg=list(train_db_pos_neg)
    print('after train_db_neg=list(train_db_pos_neg)', end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # return train_db_pos, train_db_neg
    return train_db_pos_neg

