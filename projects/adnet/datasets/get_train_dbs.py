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

def process_data_vot(train_sequences, vid_info, opt,train_db_pos,train_db_neg,lock):
    opts=opt.copy()
    train_db_pos_gpu = []
    train_db_neg_gpu = []

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

        train_db_pos_gpu.append(train_db_pos_)
        train_db_neg_gpu.append(train_db_neg_)

    try:
        lock.acquire()
        #print("len(train_db_pos_gpu): %d"%len(train_db_pos_gpu))
        train_db_pos.extend(train_db_pos_gpu)
        #print("len(train_db_pos): %d" % len(train_db_pos))
        #print("len(train_db_neg_gpu): %d" % len(train_db_neg_gpu))
        train_db_neg.extend(train_db_neg_gpu)
        #print("len(train_db_neg): %d" % len(train_db_neg))
    except Exception as err:
        raise err
    finally:
        lock.release()



def get_train_dbs(vid_info, opts):
    img = cv2.imread(vid_info['img_files'][0])

    opts['scale_factor'] = 1.05
    opts['imgSize'] = list(img.shape)
    gt_skip = opts['train']['gt_skip']

    if vid_info['db_name'] == 'alov300':
        train_sequences = vid_info['gt_use'] == 1
    else:
        train_sequences = list(range(0, vid_info['nframes'], gt_skip))

    train_db_pos = multiprocessing.Manager().list()
    train_db_neg = multiprocessing.Manager().list()
    # t0 = time.time()

    gpu_num = 27
    all_img_num=len(train_sequences)
    if all_img_num<gpu_num:
        gpu_num=all_img_num
    every_gpu_img = all_img_num // gpu_num
    img_paths_as = []
    for gn in range(gpu_num - 1):
        img_paths_as.append(train_sequences[gn * every_gpu_img:(gn + 1) * every_gpu_img])
    img_paths_as.append(train_sequences[(gpu_num - 1) * every_gpu_img:])

    lock = multiprocessing.Manager().Lock()
    record = []
    for i in range(gpu_num):
        process = multiprocessing.Process(target=process_data_vot,
                                          args=(img_paths_as[i], vid_info, opts, train_db_pos, train_db_neg, lock))
        process.start()
        record.append(process)
    for process in record:
        process.join()

    # t1 = time.time()
    # all_time = t1 - t0
    # all_m = all_time // 60
    # all_s = all_time % 60
    # print('spend time: %d m  %d s (%d s)' % (all_m, all_s, all_time))
    #print("finally: len(train_db_pos): %d" % len(train_db_pos))
    #print("finally: len(train_db_neg): %d" % len(train_db_neg))
    train_db_pos=list(train_db_pos)
    train_db_neg=list(train_db_neg)
    return train_db_pos, train_db_neg

# img_ii = 0
# box_ii = 0
# box_ii_start=0
# train_db_pos = []
# train_db_neg = []

def process_data_ILSVR(img_paths, opt,train_db_pos,train_db_neg,lock):
    opts=opt.copy()
    train_db_pos_gpu = []
    train_db_neg_gpu = []
    for train_i in img_paths:
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

        for gt_bbox in gt_bboxs:
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
            img_path='../datasets/data/ILSVRC/Data/VID/train/'+train_i+'.JPEG'

            train_db_pos_['img_path'] = np.full(len(pos_examples), img_path)
            train_db_pos_['bboxes'] = pos_examples
            train_db_pos_['labels'] = action_labels_pos
            # score labels: 1 is positive. 0 is negative
            train_db_pos_['score_labels'] = list(np.ones(len(pos_examples), dtype=int))

            train_db_neg_['img_path'] = np.full(len(neg_examples), img_path)
            train_db_neg_['bboxes'] = neg_examples
            train_db_neg_['labels'] = action_labels_neg
            # score labels: 1 is positive. 0 is negative
            train_db_neg_['score_labels'] = list(np.zeros(len(neg_examples), dtype=int))

            if len(train_db_pos_['img_path'])!=0 and len(train_db_neg_['img_path'])!=0:
                train_db_pos_gpu.append(train_db_pos_)
                train_db_neg_gpu.append(train_db_neg_)

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
        train_db_pos.extend(train_db_pos_gpu)
        # print("len(train_db_pos): %d" % len(train_db_pos))
        # print("len(train_db_neg_gpu): %d" % len(train_db_neg_gpu))
        train_db_neg.extend(train_db_neg_gpu)
        # print("len(train_db_neg): %d" % len(train_db_neg))
    except Exception as err:
        raise err
    finally:
        lock.release()
    #lock.acquire()
    #print(sign, os.getpid())
    #lock.release()

def get_train_dbs_ILSVR(opts):
    #gt_file_path = '../datasets/data/ILSVRC/Data/VID/train/ILSVRC2017_VID_train_0000/ILSVRC2017_train_00137000/000305.JPEG'
    #img = cv2.imread(gt_file_path)

    opts['scale_factor'] = 1.05
    #opts['imgSize'] = list(img.shape)
    gt_skip = opts['train']['gt_skip']

    print('before get_train_dbs_ILSVR', end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    #train_sequences = list(range(0, vid_info['nframes'], gt_skip))

    train_db_pos = multiprocessing.Manager().list()
    train_db_neg = multiprocessing.Manager().list()

    train_img_info_file=os.path.join('../datasets/data/ILSVRC/ImageSets/VID/train.txt')
    train_img_info = open(train_img_info_file, "r")
    img_paths = train_img_info.readlines()
    img_paths = img_paths[::gt_skip + 1]
    img_paths=[line.split(' ')[0] for line in img_paths]
    train_img_info.close()

    #gt_file_path = '../datasets/data/ILSVRC/Annotations/VID/train/' + img_paths[0] + '.xml'

    # img_ii = 0
    # box_ii = 0
    # box_ii_start=0
    all_img_num = len(img_paths)
    # t0=time.time()
    #t2 = time.time()

    cpu_num=27
    if all_img_num<cpu_num:
        cpu_num=all_img_num
    every_gpu_img=all_img_num//cpu_num
    img_paths_as=[]
    for gn in range(cpu_num-1):
        img_paths_as.append(img_paths[gn*every_gpu_img:(gn+1)*every_gpu_img])
    img_paths_as.append(img_paths[(cpu_num-1) * every_gpu_img:])

    lock = multiprocessing.Manager().Lock()
    record = []
    for i in range(cpu_num):
        process = multiprocessing.Process(target=process_data_ILSVR, args=(img_paths_as[i], opts,train_db_pos,train_db_neg,lock))
        process.start()
        record.append(process)
    for process in record:
        process.join()

    # t1=time.time()
    # all_time=t1-t0
    # all_m = all_time // 60
    # all_s = all_time % 60
    # print('spend time: %d m  %d s (%d s)' % (all_m, all_s, all_time))

    # print("finally: len(train_db_pos): %d" % len(train_db_pos))
    # print("finally: len(train_db_neg): %d" % len(train_db_neg))
    print('before train_db_pos=list(train_db_pos)', end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train_db_pos=list(train_db_pos)
    print('before train_db_neg=list(train_db_neg)', end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train_db_neg=list(train_db_neg)
    print('after train_db_neg=list(train_db_neg)', end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
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