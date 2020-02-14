import glob, os
import xml.etree.ElementTree as ET
import tensorflow as tf

slim = tf.contrib.slim

import cv2, glob, os, re
import numpy as np
#import scipy.io as sio
#import tracker_util as tutil

def test():
    video_infos = {
        # 'imgsize': [], #in supervised training, imgsize is used for generating boxes that near the gt box
        'gts': []
    }
    video_infos['gts'].append([0,0,0,0])
    state=video_infos['gts'][0]
    # if state == [0, 0, 0, 0]:
    #     print("debug")
    # vidDir = os.path.join('../datasets/data/ILSVRC/Data/VID/snippets/train')
    # img_files = glob.glob(os.path.join(vidDir,'ILSVRC201*_VID_train_000*','*.mp4'))
    # img_files.sort(key=str.lower)

def get_ILSVRC_videos_infos():
    '''
    get {gts,img_files(path),name,db_name,nframes}for all videos
    :param file_path: the path of the train.txt
    :return:
    '''
    videos_infos =[]
    train_videos={
        'video_names':[],
        'video_paths':[],
        #'bench_names':[]
    }
    video_infos = {
        #'imgsize': [], #in supervised training, imgsize is used for generating boxes that near the gt box
        'gt': [],
        'img_files':[],
        'nframes':0
    }
    last_video_full=True
    train_img_info_file = os.path.join('../datasets/data/ILSVRC/ImageSets/VID/train.txt')
    train_img_info = open(train_img_info_file, "r")
    img_paths = train_img_info.readlines()
    #img_paths = img_paths[::gt_skip + 1]
    img_paths = [line.split(' ')[0] for line in img_paths]
    train_img_info.close()
    for train_i in range(len(img_paths)):
        if img_paths[train_i][-6:]=='000000':
            if train_i!=0:
                if last_video_full==False:
                    last_video_full=True
                else:
                    video_infos['nframes']=int(img_paths[train_i-1][-6:])+1
                    videos_infos.append(video_infos)
                    train_videos['video_names'].append(img_paths[train_i-1][-32:-7])
                    train_videos['video_paths'].append('../datasets/data/ILSVRC/Data/VID/train/' + img_paths[train_i-1][:-32])
                    #train_videos['bench_names'] =

                    video_infos = {
                        # 'imgsize': [], #in supervised training, imgsize is used for generating boxes that near the gt box
                        'gt': [],
                        'img_files': [],
                        'nframes': 0
                    }
        elif last_video_full==False:
            continue
        gt_file_path = '../datasets/data/ILSVRC/Annotations/VID/train/' + img_paths[train_i] + '.xml'
        # gt_bbox=get_xml_box_label(gt_file_path)
        # opts['imgSize'] = get_xml_img_size(gt_file_path)
        imginfo = get_xml_img_info(gt_file_path)
        if(len(imginfo['gts'])==0):
            #print("stop")
            #imginfo['gts'].append([0,0,0,0])
            last_video_full=False
            if img_paths[train_i][-6:]!='000000':
                video_infos['nframes'] = int(img_paths[train_i - 1][-6:]) + 1
                videos_infos.append(video_infos)
                train_videos['video_names'].append(img_paths[train_i - 1][-32:-7])
                train_videos['video_paths'].append(
                    '../datasets/data/ILSVRC/Data/VID/train/' + img_paths[train_i - 1][:-32])
                video_infos = {
                    # 'imgsize': [], #in supervised training, imgsize is used for generating boxes that near the gt box
                    'gt': [],
                    'img_files': [],
                    'nframes': 0
                }
            continue
        video_infos['gt'].append(imginfo['gts'][0])
        img_path = '../datasets/data/ILSVRC/Data/VID/train/' + img_paths[train_i] + '.JPEG'
        video_infos['img_files'].append(img_path)
    video_infos['nframes'] = int(img_paths[-1][-6:]) + 1
    videos_infos.append(video_infos)
    train_videos['video_names'].append(img_paths[-1][-32:-7])
    train_videos['video_paths'].append('../datasets/data/ILSVRC/Data/VID/train/' + img_paths[-1][:-32])
    return videos_infos,train_videos


def get_xml_img_info(xmlpath):
    img_info = {
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
        bb = obj.find('bndbox')
        gt = [0, 0, 0, 0]
        gt[0] = int(bb.find('xmin').text)
        gt[1] = int(bb.find('ymin').text)
        gt[2] = int(bb.find('xmax').text)
        gt[3] = int(bb.find('ymax').text)
        gt[2] = gt[2] - gt[0]
        gt[3] = gt[3] - gt[1]

        gts.append(gt)

        # track_id=int(obj.find('trackid').text)
        # if track_id==0:
        #     break
    in_file.close()
    img_info['imgsize'] =imgsize
    img_info['gts']=gts
    return img_info

def get_xml_img_size(xmlpath):
    in_file = open(xmlpath)
    tree = ET.parse(in_file)
    root = tree.getroot()
    imgsize = [0, 0]
    siz=root.find('size')
    imgsize[1]=int(siz.find('width').text)
    imgsize[0] = int(siz.find('height').text)
    in_file.close()
    return imgsize

def get_xml_box_label(xmlpath):
    '''
    get an image's bounding box label from a .xml file
    :param xmlpath: xml file path
    :return: bounding box ground trouth
    '''

    in_file = open(xmlpath)
    tree = ET.parse(in_file)
    root = tree.getroot()
    gt = [0, 0, 0, 0]
    for obj in root.iter('object'):
        bb = obj.find('bndbox')
        #gt = [1,2,3,4]
        gt[0] = int(bb.find('xmin').text)
        gt[1] = int(bb.find('ymin').text)
        gt[2] = int(bb.find('xmax').text)
        gt[3] = int(bb.find('ymax').text)
        gt[2] = gt[2] - gt[0]
        gt[3] = gt[3] - gt[1]
        track_id=int(obj.find('trackid').text)
        if track_id==0:
            break
    in_file.close()
    return gt


def generate_vid_box_label(path_home):
    '''
    generate and save box labels(xmin,ymin,box_width,box_height) of an imagenet vid dataset video to a .txt file
    :param path_home: the path of the labels(.xml files which contains frame info and objects infos) of the video frames
    :return: saved as a .txt file
    '''
    fs = os.path.join('%s' % path_home, '*')
    out_file = open('%s.txt' % path_home, 'w')
    dataset = glob.glob(fs)
    dataset.sort()
    for p in dataset:
        in_file = open(p)
        tree = ET.parse(in_file)
        root = tree.getroot()
        for obj in root.iter('object'):
            bb = obj.find('bndbox')
            gt = [1,2,3,4]
            gt[0] = int(bb.find('xmin').text)
            gt[1] = int(bb.find('ymin').text)
            gt[2] = int(bb.find('xmax').text)
            gt[3] = int(bb.find('ymax').text)
            gt[2] = gt[2] - gt[0]
            gt[3] = gt[3] - gt[1]
            track_id=int(obj.find('trackid').text)
            if track_id==0:
                out_file.write(str(gt[0])+','+str(gt[1])+','+str(gt[2])+','+str(gt[3])+'\n')
                break
    in_file.close()
    out_file.close()

def show_gt_box(vidpath,gt_path=None):
    '''
    show the video frames with a boundbox around the object
    :param vidpath: the path of the .mp4 video file
    :param gt_path: the path of the .txt box labels(xmin,ymin,box_width,box_height)
    :return:
    '''
    if(gt_path==None):
        gt_path = vidpath[:-4] + '.txt'
    cap = cv2.VideoCapture(vidpath)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ground_truth = open(gt_path)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (640, 480))
    for f in range(length):
        success, img = cap.read()
        vid_width, vid_height = img.shape[:2]
        if f==0:
            out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (vid_height, vid_width))
        gt = ground_truth.readline()
        # min_x,min_y,box_width,box_height
        gt = np.array(re.findall('\d+', gt), dtype=int)
        frame = np.copy(img)
        frame = cv2.rectangle(frame, (int(gt[0]), int(gt[1])),
                              (int(gt[0] + gt[2]), int(gt[1] + gt[3])), [0, 0, 55], 2)

        #img_name = vidpath[7:-4] + '_%d' % f + '.jpg'
        #cv2.imwrite('results/test/' + img_name, frame) #save frames to  .jpg files
        #out.write(frame)    #save frames to a .mp4 file
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', vid_height, vid_width)
        cv2.imshow('image', frame)
        k = cv2.waitKey(1)
        # press q to exit
        # if (k & 0xff == ord('q')):
        #     break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def imgs_to_mp4(imgspath,img_extension,outpath=None):
    '''

    :param imgspath: the path of the images
    :param img_extension: the extension of img, for example, jpg/png/jpeg...
    :param outpath: the path of the mp4 file
    :return:
    '''
    if (outpath == None):
        outpath = imgspath + 'output.mp4'
    paths=glob.glob(os.path.join(imgspath, '*.%s'%img_extension))
    paths.sort(key=str.lower)

    #cap = cv2.VideoCapture(vidpath)
    #length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length=len(paths)
    #ground_truth = open(gt_path)
    fps=20.0
    out = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'MP4V'), fps, (640, 480))
    for f in range(length):
        # print('%d / %d : %s'%(f,length,paths[f]))
        #success, img = cap.read()
        if f%10 !=0:
            continue
        img=cv2.imread(paths[f])
        vid_width, vid_height = img.shape[:2]
        if f == 0:
            out = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'MP4V'), fps, (vid_height, vid_width))
        #gt = ground_truth.readline()
        # min_x,min_y,box_width,box_height
        #gt = np.array(re.findall('\d+', gt), dtype=int)
        #frame = np.copy(img)
        #frame = cv2.rectangle(frame, (int(gt[0]), int(gt[1])),
        #                      (int(gt[0] + gt[2]), int(gt[1] + gt[3])), [0, 0, 55], 2)

        # img_name = vidpath[7:-4] + '_%d' % f + '.jpg'
        # cv2.imwrite('results/test/' + img_name, frame) #save frames to  .jpg files
        #out.write(frame)    #save frames to a .mp4 file
        out.write(img)
        '''
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', vid_height, vid_width)
        cv2.imshow('image', img)
        k = cv2.waitKey(1)
        #press k to exit        
        if (k & 0xff == ord('q')):
            break
        '''
    #cap.release()
    out.release()
    #cv2.destroyAllWindows()

def do_iou_precise(path_exam,path_gt,thre=0.7):
    '''
    compute the iou and save to output/iou.txt
    print the average iou and precise
    :param path_exam:
    :param path_gt:
    :return:
    '''
    path_home="output/"
    x1 = np.load(path_exam)
    x2 = np.load(path_gt)
    from utils.overlap_ratio import overlap_ratio
    iou=overlap_ratio(x1,x2)
    np.savetxt(path_home+'iou.txt',iou,fmt='%.06f') #fmt: keep 6 numbers after dot
    iou=np.array(iou)
    average_iou=iou.mean()
    right_rs=iou>thre
    right_rs=iou[right_rs]
    precise=right_rs.size/iou.size
    print("average_iou: "+str(average_iou)+"  ;\t  precise: "+str(precise))


if __name__ == '__main__':
    #generate_vid_box_label('datasets/data/test/ILSVRC2015_train_00146003')
    #show_gt_box(vidpath='datasets/data/test/ILSVRC2015_train_00146003.mp4',gt_path='datasets/data/test/vid/ILSVRC2015_train_00146003/groundtruth.txt')
    imgs_to_mp4('/home/zb/project/detectron2/projects/adnet/mains/save_result_images/ADNet_SL_backup-0.5/ILSVRC2015_train_00146003/','jpg')
    #do_iou_precise("mains/results_on_test_images_part2/ADNet_RL_epoch29-0.5/ILSVRC2015_train_00146003-bboxes.npy","mains/results_on_test_images_part2/ADNet_RL_epoch29-0.5/ILSVRC2015_train_00146003-ground_truth.npy")
    # test()

    #get_ILSVRC_videos_infos()
    print("over")