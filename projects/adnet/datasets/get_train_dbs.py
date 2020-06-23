# matlab code:
# https://github.com/hellbell/ADNet/blob/3a7955587b5d395401ebc94a5ab067759340680d/train/get_train_dbs.m
import sys,os,time
import random
import multiprocessing
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import numpy as np
import numpy.matlib
import argparse
from utils.gen_samples import gen_samples
from utils.overlap_ratio import overlap_ratio
from utils.gen_action_labels import gen_action_labels
from utils.gen_action_labels import gen_action_pos_neg_labels
from utils.my_util import get_xml_box_label
from utils.my_util import get_xml_img_size
from utils.my_util import get_xml_img_info
from utils.my_util import get_ILSVRC_eval_infos,cal_iou
from utils.do_action import do_action

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

# def process_data_ILSVR(img_paths, opt,train_db_pos,train_db_neg,lock):
def process_data_ILSVR(img_paths, opt, train_db_pos_neg_all, lock):
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

def get_train_dbs_ILSVR(opts):
    #gt_file_path = '../datasets/data/ILSVRC/Data/VID/train/ILSVRC2017_VID_train_0000/ILSVRC2017_train_00137000/000305.JPEG'
    #img = cv2.imread(gt_file_path)

    opts['scale_factor'] = 1.05
    #opts['imgSize'] = list(img.shape)
    gt_skip = opts['train']['gt_skip']

    print('before get_train_dbs_ILSVR', end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    #train_sequences = list(range(0, vid_info['nframes'], gt_skip))

    # train_db_pos = multiprocessing.Manager().list()
    # train_db_neg = multiprocessing.Manager().list()
    train_db_pos_neg = multiprocessing.Manager().list()

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

    # cpu_num=27
    cpu_num = 24
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
        process = multiprocessing.Process(target=process_data_ILSVR, args=(img_paths_as[i], opts,train_db_pos_neg,lock))
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
    print('before train_db_pos=list(train_db_pos_neg)', end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train_db_pos_neg=list(train_db_pos_neg)
    # print('before train_db_neg=list(train_db_neg)', end=' : ')
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # train_db_neg=list(train_db_neg)
    print('after train_db_neg=list(train_db_pos_neg)', end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # return train_db_pos, train_db_neg
    return train_db_pos_neg


def process_data_ILSVR_consecutive_frame(img_paths, opt, train_db_pos_neg_all, lock):
    opts=opt.copy()
    # train_db_pos_neg_gpu = []
    train_db_pos_neg = {
        'img_path': [],  # train_i['img_files'][i],
        'bboxes': [],
        'labels': [],
        'score_labels': []
    }
    for train_i in img_paths:
        n_frames=len(train_i['gt'])
        max_dis=15
        gt_file_path = '../datasets/data/ILSVRC/Annotations/VID/train/' + train_i['img_files'][0][39:-5] + '.xml'
        imginfo = get_xml_img_info(gt_file_path)
        opts['imgSize'] = imginfo['imgsize']

        for i in range(n_frames-1,0,-1):
            # train_db_pos_neg = {
            #     'img_path': train_i['img_files'][i],
            #     'bboxes': [],
            #     'labels': [],
            #     'score_labels': []
            # }
            # del_t=len(train_i['trackid'][i])
            # if del_t>1:
            #     print("debug")
            for l in range(len(train_i['trackid'][i])):
                gt_bbox = train_i['gt'][i][l]
                # train_db_pos_neg = {
                #     'img_path': [],#train_i['img_files'][i],
                #     'bboxes': [],
                #     'labels': [],
                #     'score_labels': []
                # }
                bk_sign=False
                for j in range(i-1,i-max_dis-1,-1):
                    if j<0:
                        break
                    for k in range(len(train_i['trackid'][j])):

                        if train_i['trackid'][j][k]==train_i['trackid'][i][l]:
                            # train_db_pos_neg = {
                            #     'img_path': train_i['img_files'][i],
                            #     'bboxes': [],
                            #     'labels': [],
                            #     'score_labels': []
                            # }
                            pos_neg_box=train_i['gt'][j][k]
                            c_iou=cal_iou(pos_neg_box,gt_bbox)
                            # del_iou=cal_iou(pos_neg_box,gt_bbox)
                            # print(i-j,del_iou)
                            if c_iou>0.7:
                                action_label_pos, _ = gen_action_pos_neg_labels(opts['num_actions'], opts,
                                                                                               np.array(pos_neg_box),
                                                                                               gt_bbox)




                                train_db_pos_neg['img_path'].append(train_i['img_files'][i])
                                train_db_pos_neg['bboxes'].append(pos_neg_box)
                                action_label_pos = np.transpose(action_label_pos).tolist()
                                train_db_pos_neg['labels'].extend(action_label_pos)
                                train_db_pos_neg['score_labels'].extend(list(np.ones(1, dtype=int)))
                                # train_db_pos_neg_gpu.append(train_db_pos_neg)
                            else:
                                bk_sign=True
                                break

                            # train_db_pos_neg = {
                            #     'img_path': train_i['img_files'][i],
                            #     'bboxes': [],
                            #     'labels': [],
                            #     'score_labels': []
                            # }
                            if (i-j)%3==0:
                                nct = -1
                                while True:
                                    # in original code, this 1 line below use opts['nPos_train'] instead of opts['nNeg_train']
                                    nct += 1
                                    if nct == 20:
                                        break
                                    neg = gen_samples('gaussian', gt_bbox, 5, opts, 2, 10)
                                    r = overlap_ratio(neg, np.matlib.repmat(gt_bbox, len(neg), 1))
                                    # neg = neg[np.array(r) < opts['consecutive_negThre_train']]
                                    neg = neg[np.array(r) < opts['consecutive_negThre_train']]
                                    if len(neg) == 0:
                                        continue
                                        # break
                                    else:
                                        pos_neg_box = neg[0]
                                        # print("neg[0]", end=": ")
                                        # print(neg[0])
                                        break
                                train_db_pos_neg['img_path'].append(train_i['img_files'][i])
                                train_db_pos_neg['bboxes'].append(pos_neg_box)
                                action_label_neg = np.full((opts['num_actions'], 1), fill_value=-1)
                                action_label_neg = np.transpose(action_label_neg).tolist()
                                train_db_pos_neg['labels'].extend(action_label_neg)
                                train_db_pos_neg['score_labels'].extend(list(np.zeros(1, dtype=int)))
                            # train_db_pos_neg_gpu.append(train_db_pos_neg)
                    if bk_sign==True:
                        break

                # if len(train_db_pos_neg['bboxes']) >0:
                # if len(train_db_pos_neg['bboxes']) == 20:
                #     train_db_pos_neg_gpu.append(train_db_pos_neg)
    try:
        lock.acquire()
        # train_db_pos_neg_all.extend(train_db_pos_neg_gpu)
        train_db_pos_neg_all.append(train_db_pos_neg)
    except Exception as err:
        raise err
    finally:
        lock.release()

def get_train_dbs_ILSVR_consecutive_frame(opts):
    #gt_file_path = '../datasets/data/ILSVRC/Data/VID/train/ILSVRC2017_VID_train_0000/ILSVRC2017_train_00137000/000305.JPEG'
    #img = cv2.imread(gt_file_path)

    opts['scale_factor'] = 1.05
    #opts['imgSize'] = list(img.shape)
    gt_skip = opts['train']['gt_skip']

    print('before get_train_dbs_ILSVR_consecutive_frame', end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_imgs', default=0, type=int,
                        help='the num of imgs that picked from val.txt, 0 represent all imgs')
    parser.add_argument('--gt_skip', default=1, type=int, help='frame sampling frequency')
    parser.add_argument('--dataset_year', default=2222, type=int, help='dataset version, like ILSVRC2015, ILSVRC2017, 2222 means train.txt')
    args2 = parser.parse_args(['--eval_imgs','0','--gt_skip','1','--dataset_year','2222'])

    videos_infos, _ = get_ILSVRC_eval_infos(args2)
    print('before process_data_ILSVR_consecutive_frame', end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    #train_sequences = list(range(0, vid_info['nframes'], gt_skip))

    # train_db_pos = multiprocessing.Manager().list()
    # train_db_neg = multiprocessing.Manager().list()
    train_db_pos_neg = multiprocessing.Manager().list()

    # train_img_info_file=os.path.join('../datasets/data/ILSVRC/ImageSets/VID/train.txt')
    # train_img_info = open(train_img_info_file, "r")
    # img_paths = train_img_info.readlines()
    # img_paths = img_paths[::gt_skip + 1]
    # img_paths=[line.split(' ')[0] for line in img_paths]
    # train_img_info.close()

    #gt_file_path = '../datasets/data/ILSVRC/Annotations/VID/train/' + img_paths[0] + '.xml'

    # img_ii = 0
    # box_ii = 0
    # box_ii_start=0
    all_img_num = len(videos_infos)
    # t0=time.time()
    #t2 = time.time()

    # cpu_num=27
    cpu_num = 24
    if all_img_num<cpu_num:
        cpu_num=all_img_num
    every_gpu_img=all_img_num//cpu_num
    img_paths_as=[]
    for gn in range(cpu_num-1):
        img_paths_as.append(videos_infos[gn*every_gpu_img:(gn+1)*every_gpu_img])
    img_paths_as.append(videos_infos[(cpu_num-1) * every_gpu_img:])

    lock = multiprocessing.Manager().Lock()
    record = []
    for i in range(cpu_num):
        process = multiprocessing.Process(target=process_data_ILSVR_consecutive_frame, args=(img_paths_as[i], opts,train_db_pos_neg,lock))
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
    print('before train_db_pos=list(train_db_pos_neg)', end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train_db_pos_neg=list(train_db_pos_neg)
    # print('before train_db_neg=list(train_db_neg)', end=' : ')
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # train_db_neg=list(train_db_neg)
    print('after train_db_neg=list(train_db_pos_neg)', end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # return train_db_pos, train_db_neg
    return train_db_pos_neg

#random choose the highest iou steps
def process_data_mul_step(img_paths, opt, train_db_pos_neg_all, lock):
    opts=opt.copy()
    # train_db_pos_neg_gpu = []
    train_db_pos_neg = {
        'img_path': [],  # train_i['img_files'][i],
        'bboxes': [],
        'labels': [],
        'score_labels': []
    }
    distan=1
    for train_i in img_paths:
        n_frames=len(train_i['gt'])
        # max_dis=15
        gt_file_path = '../datasets/data/ILSVRC/Annotations/VID/train/' + train_i['img_files'][0][39:-5] + '.xml'
        imginfo = get_xml_img_info(gt_file_path)
        opts['imgSize'] = imginfo['imgsize']

        for i in range(0,n_frames-distan-1,5):
            for l in range(len(train_i['trackid'][i])):
                # train_db_pos_neg = {
                #     'img_path': train_i['img_files'][i + distan],
                #     'bboxes': [],
                #     'labels': [],
                #     'score_labels': []
                # }
                for k in range(len(train_i['trackid'][i + distan])):
                    if train_i['trackid'][i][l] == train_i['trackid'][i + distan][k]:
                        gt_end = train_i['gt'][i + distan][k]
                iou_max=0
                step_max=[]
                box_max=[]
                for lp in range(500):
                    curr_bbox = train_i['gt'][i][l]
                    step=[]
                    box=[]
                    for st in range(5): #step numbers
                        action=random.randint(0, 10)
                        # if st==0:
                        #     print(action)
                        step.append(action)
                        box.append(curr_bbox)
                        curr_bbox = do_action(curr_bbox, opts, action, opts['imgSize'])
                    box.append(curr_bbox)
                    step.append(opts['stop_action'])  #stop action
                    # c_iou=cal_iou(curr_bbox,gt_end)
                    t_iou_max=cal_iou(curr_bbox,gt_end)
                    t_max_n=-1
                    for st in range(5):
                        t_iou=cal_iou(box[st],gt_end)
                        if t_iou>t_iou_max:
                            t_iou_max=t_iou
                            t_max_n=st
                    if t_max_n>-1:
                        box=box[:t_max_n+1]
                        step=step[:t_max_n]
                        step.append(opts['stop_action'])
                    if t_iou_max>iou_max:
                        iou_max=t_iou_max
                        step_max=step
                        box_max=box
                if iou_max>opts['stopIou']:  #save data to train_db
                    for datai in range(len(step_max)):
                        train_db_pos_neg['img_path'].append(train_i['img_files'][i+distan])
                        train_db_pos_neg['bboxes'].append(box_max[datai])
                        action_t = np.zeros(opts['num_actions'])
                        action_t[step_max[datai]] = 1
                        action_label_pos=action_t.tolist()
                        train_db_pos_neg['labels'].append(action_label_pos)
                        train_db_pos_neg['score_labels'].extend(list(np.ones(1, dtype=int)))

                        if (datai)%3==0:
                            nct = -1
                            while True:
                                # in original code, this 1 line below use opts['nPos_train'] instead of opts['nNeg_train']
                                nct += 1
                                if nct == 20:
                                    break
                                neg = gen_samples('gaussian', gt_end, 5, opts, 2, 10)
                                r = overlap_ratio(neg, np.matlib.repmat(gt_end, len(neg), 1))
                                # neg = neg[np.array(r) < opts['consecutive_negThre_train']]
                                neg = neg[np.array(r) < opts['consecutive_negThre_train']]
                                if len(neg) == 0:
                                    continue
                                    # break
                                else:
                                    pos_neg_box = neg[0]
                                    # print("neg[0]", end=": ")
                                    # print(neg[0])
                                    break
                            train_db_pos_neg['img_path'].append(train_i['img_files'][i+distan])
                            train_db_pos_neg['bboxes'].append(pos_neg_box)
                            action_label_neg = np.full((opts['num_actions'], 1), fill_value=-1)
                            action_label_neg = np.transpose(action_label_neg).tolist()
                            train_db_pos_neg['labels'].extend(action_label_neg)
                            train_db_pos_neg['score_labels'].extend(list(np.zeros(1, dtype=int)))
                        # train_db_pos_neg_gpu.append(train_db_pos_neg)

                # if len(train_db_pos_neg['bboxes']) >0:
                # print(iou_max,len(train_db_pos_neg['bboxes']))
                # if len(train_db_pos_neg['bboxes']) == 20:
                #     train_db_pos_neg_gpu.append(train_db_pos_neg)
    try:
        lock.acquire()
        # train_db_pos_neg_all.extend(train_db_pos_neg_gpu)
        train_db_pos_neg_all.append(train_db_pos_neg)
    except Exception as err:
        raise err
    finally:
        lock.release()


#choose the highest iou steps
def process_data_mul_step_3(img_paths, opt, train_db_pos_neg_all, lock):
    opts=opt.copy()
    # train_db_pos_neg_gpu = []
    train_db_pos_neg = {
        'img_path': [],  # train_i['img_files'][i],
        'bboxes': [],
        'labels': [],
        'score_labels': []
    }
    distan=1
    for train_i in img_paths:
        n_frames=len(train_i['gt'])
        # max_dis=15
        gt_file_path = '../datasets/data/ILSVRC/Annotations/VID/train/' + train_i['img_files'][0][39:-5] + '.xml'
        imginfo = get_xml_img_info(gt_file_path)
        opts['imgSize'] = imginfo['imgsize']

        for i in range(0,n_frames-distan-1,5):
            for l in range(len(train_i['trackid'][i])):
                # train_db_pos_neg = {
                #     'img_path': train_i['img_files'][i + distan],
                #     'bboxes': [],
                #     'labels': [],
                #     'score_labels': []
                # }
                for k in range(len(train_i['trackid'][i + distan])):
                    if train_i['trackid'][i][l] == train_i['trackid'][i + distan][k]:
                        gt_end = train_i['gt'][i + distan][k]
                iou_max=0
                step_max=[]
                box_max=[]
                curr_bbox = train_i['gt'][i][l]
                # if i==5:
                #     print("debug")
                for st in range(15):
                    box_max.append(curr_bbox)
                    t_iou_max=0
                    t_box_max=[]
                    t_act_max=-1
                    for action in range(11):
                        curr_bbox_t = do_action(curr_bbox, opts, action, opts['imgSize'])
                        t_iou = cal_iou(curr_bbox_t, gt_end)
                        if action == opts['stop_action']:
                            t_iou_act_stop = t_iou
                            t_box_act_stop = curr_bbox_t
                        if t_iou>t_iou_max:
                            t_iou_max=t_iou
                            t_act_max=action
                            t_box_max=curr_bbox_t
                    if abs(t_iou_act_stop - t_iou_max) < 0.005 and t_act_max != opts['stop_action']:
                        t_iou_max = t_iou_act_stop
                        t_act_max = opts['stop_action']
                        t_box_max = t_box_act_stop
                    if t_act_max==-1:
                        break
                    iou_max=t_iou_max
                    # if st==0:
                    #     print("")
                    #     print("start iou: %f,"%(t_iou_act_stop),end='  ')
                    # print("do %d -> %f,"%(t_act_max,iou_max),end='  ')
                    if t_act_max==opts['stop_action']:
                        step_max.append(opts['stop_action'])
                        break
                    else:
                        step_max.append(t_act_max)
                        curr_bbox=t_box_max

                # for lp in range(500):
                #     curr_bbox = train_i['gt'][i][l]
                #     step=[]
                #     box=[]
                #     for st in range(5): #step numbers
                #         action=random.randint(0, 10)
                #         # if st==0:
                #         #     print(action)
                #         step.append(action)
                #         box.append(curr_bbox)
                #         curr_bbox = do_action(curr_bbox, opts, action, opts['imgSize'])
                #     box.append(curr_bbox)
                #     step.append(opts['stop_action'])  #stop action
                #     # c_iou=cal_iou(curr_bbox,gt_end)
                #     t_iou_max=cal_iou(curr_bbox,gt_end)
                #     t_max_n=-1
                #     for st in range(5):
                #         t_iou=cal_iou(box[st],gt_end)
                #         if t_iou>t_iou_max:
                #             t_iou_max=t_iou
                #             t_max_n=st
                #     if t_max_n>-1:
                #         box=box[:t_max_n+1]
                #         step=step[:t_max_n]
                #         step.append(opts['stop_action'])
                #     if t_iou_max>iou_max:
                #         iou_max=t_iou_max
                #         step_max=step
                #         box_max=box
                if iou_max>opts['stopIou']:  #save data to train_db
                    for datai in range(len(step_max)):
                        train_db_pos_neg['img_path'].append(train_i['img_files'][i+distan])
                        train_db_pos_neg['bboxes'].append(box_max[datai])
                        action_t = np.zeros(opts['num_actions'])
                        action_t[step_max[datai]] = 1
                        action_label_pos=action_t.tolist()
                        train_db_pos_neg['labels'].append(action_label_pos)
                        train_db_pos_neg['score_labels'].extend(list(np.ones(1, dtype=int)))

                        if (datai)%3==0:
                            nct = -1
                            while True:
                                # in original code, this 1 line below use opts['nPos_train'] instead of opts['nNeg_train']
                                nct += 1
                                if nct == 20:
                                    break
                                neg = gen_samples('gaussian', gt_end, 5, opts, 2, 10)
                                r = overlap_ratio(neg, np.matlib.repmat(gt_end, len(neg), 1))
                                # neg = neg[np.array(r) < opts['consecutive_negThre_train']]
                                neg = neg[np.array(r) < opts['consecutive_negThre_train']]
                                if len(neg) == 0:
                                    continue
                                    # break
                                else:
                                    pos_neg_box = neg[0]
                                    # print("neg[0]", end=": ")
                                    # print(neg[0])
                                    break
                            train_db_pos_neg['img_path'].append(train_i['img_files'][i+distan])
                            train_db_pos_neg['bboxes'].append(pos_neg_box)
                            action_label_neg = np.full((opts['num_actions'], 1), fill_value=-1)
                            action_label_neg = np.transpose(action_label_neg).tolist()
                            train_db_pos_neg['labels'].extend(action_label_neg)
                            train_db_pos_neg['score_labels'].extend(list(np.zeros(1, dtype=int)))
                        # train_db_pos_neg_gpu.append(train_db_pos_neg)

                # if len(train_db_pos_neg['bboxes']) >0:
                # print(iou_max,len(train_db_pos_neg['bboxes']))
                # if len(train_db_pos_neg['bboxes']) == 20:
                #     train_db_pos_neg_gpu.append(train_db_pos_neg)
    try:
        lock.acquire()
        # train_db_pos_neg_all.extend(train_db_pos_neg_gpu)
        train_db_pos_neg_all.append(train_db_pos_neg)
    except Exception as err:
        raise err
    finally:
        lock.release()

#teacher Zhong: choose every first step of a loop to combine to a step list
def process_data_mul_step_2(img_paths, opt, train_db_pos_neg_all, lock):
    opts=opt.copy()
    train_db_pos_neg_gpu = []
    for train_i in img_paths:
        n_frames=len(train_i['gt'])
        # max_dis=15
        gt_file_path = '../datasets/data/ILSVRC/Annotations/VID/train/' + train_i['img_files'][0][39:-5] + '.xml'
        imginfo = get_xml_img_info(gt_file_path)
        opts['imgSize'] = imginfo['imgsize']

        for i in range(0,n_frames-2,5):
            for l in range(len(train_i['trackid'][i])):
                train_db_pos_neg = {
                    'img_path': train_i['img_files'][i + 1],
                    'bboxes': [],
                    'labels': [],
                    'score_labels': []
                }
                for k in range(len(train_i['trackid'][i + 1])):
                    if train_i['trackid'][i][l] == train_i['trackid'][i + 1][k]:
                        gt_end = train_i['gt'][i + 1][k]
                
                step_list=[]
                box_list=[]
                box_list.append(train_i['gt'][i][l])
                for st_list in range(14):
                    iou_max=-1
                    step_max=[]
                    box_max=[]
                    for lp in range(50):
                        curr_bbox = box_list[-1]
                        step=[]
                        box=[]
                        for st in range(5): #step numbers
                            action=random.randint(0, 10)
                            step.append(action)
                            box.append(curr_bbox)
                            curr_bbox = do_action(curr_bbox, opts, action, opts['imgSize'])
                        box.append(curr_bbox)
                        step.append(opts['stop_action'])  #stop action
                        c_iou=cal_iou(curr_bbox,gt_end)
                        if c_iou>iou_max:
                            iou_max=c_iou
                            step_max=step
                            box_max=box
                    # if len(step_max)==0:
                    #     print(c_iou,iou_max)
                    step_list.append(step_max[0])
                    box_list.append(box_max[1])
                step_list.append(opts['stop_action'])
                iou_max=cal_iou(box_list[-1],gt_end)
                if iou_max>opts['stopIou']:  #save data to train_db
                    for datai in range(len(step_list)):
                        train_db_pos_neg['bboxes'].append(box_list[datai])
                        action_t = np.zeros(opts['num_actions'])
                        action_t[step_list[datai]] = 1
                        action_label_pos=action_t.tolist()
                        train_db_pos_neg['labels'].append(action_label_pos)
                        train_db_pos_neg['score_labels'].extend(list(np.ones(1, dtype=int)))

                        if (datai)%3==0:
                            nct = -1
                            while True:
                                # in original code, this 1 line below use opts['nPos_train'] instead of opts['nNeg_train']
                                nct += 1
                                if nct == 20:
                                    break
                                neg = gen_samples('gaussian', gt_end, 5, opts, 2, 10)
                                r = overlap_ratio(neg, np.matlib.repmat(gt_end, len(neg), 1))
                                # neg = neg[np.array(r) < opts['consecutive_negThre_train']]
                                neg = neg[np.array(r) < opts['consecutive_negThre_train']]
                                if len(neg) == 0:
                                    continue
                                    # break
                                else:
                                    pos_neg_box = neg[0]
                                    # print("neg[0]", end=": ")
                                    # print(neg[0])
                                    break
                            train_db_pos_neg['bboxes'].append(pos_neg_box)
                            action_label_neg = np.full((opts['num_actions'], 1), fill_value=-1)
                            action_label_neg = np.transpose(action_label_neg).tolist()
                            train_db_pos_neg['labels'].extend(action_label_neg)
                            train_db_pos_neg['score_labels'].extend(list(np.zeros(1, dtype=int)))
                        # train_db_pos_neg_gpu.append(train_db_pos_neg)

                # if len(train_db_pos_neg['bboxes']) >0:
                # print(iou_max,len(train_db_pos_neg['bboxes']))
                if len(train_db_pos_neg['bboxes']) == 20:
                    train_db_pos_neg_gpu.append(train_db_pos_neg)
    try:
        lock.acquire()
        train_db_pos_neg_all.extend(train_db_pos_neg_gpu)
    except Exception as err:
        raise err
    finally:
        lock.release()

def get_train_dbs_mul_step(opts):

    opts['scale_factor'] = 1.05
    #opts['imgSize'] = list(img.shape)
    gt_skip = opts['train']['gt_skip']

    print('before get_train_dbs_ILSVR_consecutive_frame', end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_imgs', default=0, type=int,
                        help='the num of imgs that picked from val.txt, 0 represent all imgs')
    parser.add_argument('--gt_skip', default=1, type=int, help='frame sampling frequency')
    parser.add_argument('--dataset_year', default=2222, type=int, help='dataset version, like ILSVRC2015, ILSVRC2017, 2222 means train.txt')
    args2 = parser.parse_args(['--eval_imgs','0','--gt_skip','1','--dataset_year','2222'])

    videos_infos, _ = get_ILSVRC_eval_infos(args2)
    print('before process_data_mul_step', end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train_db_pos_neg = multiprocessing.Manager().list()
    all_img_num = len(videos_infos)

    # cpu_num=27
    cpu_num = 24
    if all_img_num<cpu_num:
        cpu_num=all_img_num
    every_gpu_img=all_img_num//cpu_num
    img_paths_as=[]
    for gn in range(cpu_num-1):
        img_paths_as.append(videos_infos[gn*every_gpu_img:(gn+1)*every_gpu_img])
    img_paths_as.append(videos_infos[(cpu_num-1) * every_gpu_img:])

    lock = multiprocessing.Manager().Lock()
    record = []
    for i in range(cpu_num):
        process = multiprocessing.Process(target=process_data_mul_step, args=(img_paths_as[i], opts,train_db_pos_neg,lock))
        process.start()
        record.append(process)
    for process in record:
        process.join()

    print('before train_db_pos=list(train_db_pos_neg)', end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    train_db_pos_neg=list(train_db_pos_neg)
    print('after train_db_neg=list(train_db_pos_neg)', end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    return train_db_pos_neg

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