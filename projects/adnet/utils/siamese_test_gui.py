import argparse
import sys
import os,time
import re 
import multiprocessing
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import _init_paths
from options.general2 import opts
from utils.gen_samples import gen_samples
from utils.display import draw_box,draw_box_bigline
from utils.augmentations import ADNet_Augmentation3
from utils.do_action import do_action
import torch
import numpy as np
torch.multiprocessing.set_start_method('spawn', force=True)
import cv2
import random
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QApplication, QWidget

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



class Thread_track(QThread):  
    _signal =pyqtSignal(str,str,str,list,int)
    def __init__(self,videos_infos,transform3,transform,siamesenet,net):
        super().__init__()
        self.videos_infos=videos_infos
        self.transform3=transform3
        self.transform=transform
        self.siamesenet=siamesenet
        self.net=net

        self.freshcheckBox=True
        self.vidx1=0
        self.f1=0
        self.tid1=0
        self.f2=0

    def run(self):
        path1 = self.videos_infos[self.vidx1]['img_files'][self.f1]
        frame1 = cv2.imread(path1)
        gt1 = self.videos_infos[self.vidx1]['gt'][self.f1][self.tid1]

        t_aera1, _, _ = self.transform3(frame1, gt1)
        curr_bbox=gt1
        if self.freshcheckBox:
            for fi2 in range(self.f1+1,self.f2+1):
                path2 = self.videos_infos[self.vidx1]['img_files'][fi2]
                frame2 = cv2.imread(path2)
                curr_bboxs, curr_scores=self.adnet_inference(frame2, curr_bbox) #look
                curr_bbox=curr_bboxs[-1]
                #self.label_path2.setText(path2)
                for ti in range(len(curr_scores)):
                    t_aera2, _, _ = self.transform3(frame2, curr_bboxs[ti])
                    output1, output2 = self.siamesenet(Variable(t_aera1).cuda(), Variable(t_aera2).cuda())
                    euclidean_distance = F.pairwise_distance(output1, output2)
                    sia_value = round(euclidean_distance.item(), 2)
                    sia_value=str(sia_value)
                    category_name2 = "step: %d/%d, score: %.2f" % (ti,len(curr_scores)-1,curr_scores[ti])
                    
                    # time.sleep(0.5)
                    if not self.freshcheckBox:
                        self._signal.emit(path2,sia_value,category_name2,curr_bboxs[ti],0)
                        return
                    else:
                        self._signal.emit(path2,sia_value,category_name2,curr_bboxs[ti],1)
        else:
            for fi2 in range(self.f1+1,self.f2+1):
                path2 = self.videos_infos[self.vidx1]['img_files'][fi2]
                frame2 = cv2.imread(path2)
                curr_bboxs, curr_scores=self.adnet_inference(frame2, curr_bbox)
                curr_bbox=curr_bboxs[-1]
            t_aera2, _, _ = self.transform3(frame2, curr_bbox)
            output1, output2 = self.siamesenet(Variable(t_aera1).cuda(), Variable(t_aera2).cuda())
            euclidean_distance = F.pairwise_distance(output1, output2)
            sia_value = round(euclidean_distance.item(), 2)
            category_name2 = "score: %.2f"%(curr_scores[-1])
            self._signal.emit(path2,sia_value,category_name2,curr_bbox,0)

    def adnet_inference(self,frame,curr_bbox):
        frame2 = frame.copy()
        frame2 = frame2.astype(np.float32)
        frame2 = torch.from_numpy(frame2).cuda()

        t = 0
        curr_bboxs=[]
        curr_scores=[]
        while True:
            curr_patch, curr_bbox, _, _ = self.transform(frame2, curr_bbox, None, None)
            fc6_out, fc7_out = self.net.forward(curr_patch)
            curr_score = fc7_out.detach().cpu().numpy()[0][1]
            curr_scores.append(curr_score)
            action = np.argmax(fc6_out.detach().cpu().numpy())
            curr_bbox = do_action(curr_bbox, opts, action, frame.shape)
            # bound the curr_bbox size
            if curr_bbox[2] < 10:
                curr_bbox[0] = min(0, curr_bbox[0] + curr_bbox[2] / 2 - 10 / 2)
                curr_bbox[2] = 10
            if curr_bbox[3] < 10:
                curr_bbox[1] = min(0, curr_bbox[1] + curr_bbox[3] / 2 - 10 / 2)
                curr_bbox[3] = 10
            curr_bboxs.append(curr_bbox)
            t += 1
            if action == opts['stop_action'] or t >= opts['num_action_step_max']:
                break
        return curr_bboxs,curr_scores

        
        
class siamese_test(QWidget):
    def __init__(self,siamesenet,net,videos_infos,transform3,transform):
        super(siamese_test, self).__init__()
        self.transform3 = transform3
        self.transform = transform
        self.videos_infos = videos_infos
        self.siamesenet = siamesenet
        self.net = net
        self.thread_track = Thread_track(videos_infos,transform3,transform,siamesenet,net)
        self.thread_track._signal.connect(self.call_back_track)
        self.initUI()

    def initUI(self):
        ft=QFont("Roman times", 20, QFont.Bold)
        pe = QPalette()
        pe.setColor(QPalette.WindowText, Qt.red)

        self.resize(1300, 650)
        self.setFixedSize(1300, 650)
        self.center()
        self.setWindowTitle("siamese result test")
        grid = QGridLayout()

        self.setLayout(grid)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 7)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 7)
        grid.setColumnStretch(4, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1,0.6)
        grid.setRowStretch(2,6)
        grid.setRowStretch(3,1)
        grid.setRowStretch(4,1)
        grid.setRowStretch(5,0.5)

        path_input_tip1 = QLabel("input_path1:")
        path_input_tip2 = QLabel("input_path2:")
        frameid_tip1 = QLabel("frameid1:")
        frameid_tip2 = QLabel("frameid2:")
        trackid_tip1 = QLabel("objid1:")
        trackid_tip2 = QLabel("objid2:")

        self.input_path1=QLineEdit(self)
        self.input_path1.setFixedSize(200, 30)
        self.input_path1.setPlaceholderText("eg: ILSVRC2015_val_00000001")
        # self.input_path1.isClearButtonEnabled()
        self.input_path2 = QLineEdit(self)
        self.input_path2.setFixedSize(200, 30)
        self.input_path2.setPlaceholderText("eg: ILSVRC2015_val_00000001")
        # self.input_path2.isClearButtonEnabled()
        self.frameid1 = QLineEdit(self)
        self.frameid1.setFixedSize(45, 30)
        self.frameid1.setPlaceholderText("eg:12")
        # self.frameid1.isClearButtonEnabled()
        self.frameid2 = QLineEdit(self)
        self.frameid2.setFixedSize(45, 30)
        self.frameid2.setPlaceholderText("eg:15")
        # self.frameid2.isClearButtonEnabled()
        self.trackid1 = QLineEdit(self)
        self.trackid1.setFixedSize(40, 30)
        self.trackid1.setPlaceholderText("eg:0")
        # self.trackid1.isClearButtonEnabled()
        self.trackid2 = QLineEdit(self)
        self.trackid2.setFixedSize(40, 30)
        self.trackid2.setPlaceholderText("eg:0")
        # self.trackid2.isClearButtonEnabled()

        button_start = QPushButton("start")
        button_start.setFont(ft)
        button_start.setFixedSize(80, 30)
        button_start.clicked.connect(self.custom_siam)

        self.freshcheckBox = QtWidgets.QCheckBox("show all")

        self.button_track = QPushButton("track&siamese")
        # button_track.setFont(ft)
        self.button_track.setFixedSize(80, 30)
        self.button_track.clicked.connect(self.track_and_siamese)

        hbox0 = QHBoxLayout()
        # hbox0.addStretch(1)
        hbox0.addWidget(path_input_tip1)
        hbox0.addWidget(self.input_path1)
        # hbox0.addStretch(1)
        hbox0.addWidget(frameid_tip1)
        hbox0.addWidget(self.frameid1)
        # hbox0.addStretch(1)
        hbox0.addWidget(trackid_tip1)
        hbox0.addWidget(self.trackid1)
        hbox0.addStretch(2)
        hbox0.addWidget(path_input_tip2)
        hbox0.addWidget(self.input_path2)
        # hbox0.addStretch(1)
        hbox0.addWidget(frameid_tip2)
        hbox0.addWidget(self.frameid2)
        # hbox0.addStretch(1)
        hbox0.addWidget(trackid_tip2)
        hbox0.addWidget(self.trackid2)
        hbox0.addStretch(1)
        hbox0.addWidget(button_start)
        hbox0.addStretch(1)
        hbox0.addWidget(self.freshcheckBox)
        hbox0.addWidget(self.button_track)
        # hbox0.addStretch(1)
        hwg = QtWidgets.QWidget()
        hwg.setLayout(hbox0)
        # hwg.setSpacing(10)
        grid.addWidget(hwg, 0, 0, 1, 6)

        label1 = QLabel("path1: ")
        # label1.setFont(ft)
        # label1.setText("path1")
        grid.addWidget(label1, 1, 0, Qt.AlignRight)
        label2 = QLabel("path2: ")
        # label2.setFont(ft)
        grid.addWidget(label2, 1, 2, Qt.AlignRight)
        label3 = QLabel("siamese distance of img1 and img2: ")
        label3.setFont(ft)
        grid.addWidget(label3, 3, 1,1,2,Qt.AlignRight)
        self.label4 = QLabel("realtime diversity")   #siamese value
        self.label4.setFont(ft)
        self.label4.setPalette(pe)
        grid.addWidget(self.label4, 3, 3)

        self.label_path1 = QLabel("realtime\\path\\to\\img1")
        self.label_path1.setPalette(pe)
        self.label_path1.setTextInteractionFlags(Qt.TextSelectableByMouse)
        grid.addWidget(self.label_path1, 1, 1)
        self.label_path2 = QLabel("realtime\\path\\to\\img2")
        self.label_path2.setPalette(pe)
        self.label_path2.setTextInteractionFlags(Qt.TextSelectableByMouse)
        grid.addWidget(self.label_path2, 1, 3)

        self.pic1=QLabel("image1 area")
        self.pic1.setAlignment(Qt.AlignCenter)
        self.pic1.setFont(ft)
        self.pic1.setPalette(pe)
        # self.pic1.setStyleSheet("border: 2px solid red")
        self.pic1.setStyleSheet("background: yellow")
        self.pic1.setFixedSize(495, 330)
        # self.pic1.setScaledContents(True)
        grid.addWidget(self.pic1, 2, 1)
        self.pic2 = QLabel("image2 area")
        self.pic2.setAlignment(Qt.AlignCenter)
        self.pic2.setFont(ft)
        self.pic2.setPalette(pe)
        self.pic2.setStyleSheet("background: yellow")
        self.pic2.setFixedSize(495, 330)
        # self.pic2.setScaledContents(True)
        grid.addWidget(self.pic2, 2, 3)

        button1 = QPushButton("random positive")
        button1.setFont(ft)
        button1.setFixedSize(280,60)
        # grid.addWidget(button1, 3,0,1,2)
        button1.clicked.connect(self.rand_pos)
        button2 = QPushButton("random negative")
        button2.setFont(ft)
        button2.setFixedSize(280,60)
        # grid.addWidget(button2, 3, 3)
        button2.clicked.connect(self.rand_neg)
        button3 = QPushButton("ramdom same frame")
        button3.setFont(ft)
        button3.setFixedSize(320, 60)
        # grid.addWidget(button3, 3, 3)
        button3.clicked.connect(self.rand_neg_same_frame)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(button1)
        hbox.addStretch(1)
        hbox.addWidget(button3)
        hbox.addStretch(1)
        hbox.addWidget(button2)
        hbox.addStretch(1)
        hwg = QtWidgets.QWidget()
        hwg.setLayout(hbox)
        # hwg.setSpacing(10)
        grid.addWidget(hwg, 4, 1,1,3)
        grid.setSpacing(5)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def track_and_siamese(self):
        p1 = self.input_path1.text()
        # p2 = self.input_path2.text()
        f1 = self.frameid1.text()
        f2 = self.frameid2.text()
        tid1 = self.trackid1.text()
        # tid2 = self.trackid2.text()

        isint = '^-?[0-9]\d*$'
        rr1 = re.compile(isint)
        if rr1.match(f1) is None:
            QMessageBox.information(self, "message box", "Please enter an integer in the 'frameid1'",
                                    QMessageBox.Yes)
            return
        if rr1.match(f2) is None:
            QMessageBox.information(self, "message box", "Please enter an integer in the 'frameid2'",
                                    QMessageBox.Yes)
            return
        if rr1.match(tid1) is None:
            QMessageBox.information(self, "message box", "Please enter an integer in the 'objid1'",
                                    QMessageBox.Yes)
            return
        # if rr1.match(tid2) is None:
        #     QMessageBox.information(self, "message box", "Please enter an integer in the 'objid2'",
        #                             QMessageBox.Yes)
        #     return

        f1 = int(f1)
        f2 = int(f2)
        tid1 = int(tid1)
        # tid2 = int(tid2)
        vlen = len(self.videos_infos)
        videos_infos = self.videos_infos
        vidx1 = -1
        vidx2 = -1

        for i in range(vlen):
            if videos_infos[i]['img_files'][0][-35:-12] == p1:
                vidx1 = i
                break
        if vidx1 == -1:
            QMessageBox.information(self, "message box", "input_path1 doesn't exist, please retype!",
                                    QMessageBox.Yes)
            return
        # if p1 == p2:
        #     vidx2 = vidx1
        # else:
        #     for i in range(vlen):
        #         if videos_infos[i]['img_files'][0][-35:-12] == p2:
        #             vidx2 = i
        #             break
        #     if vidx2 == -1:
        #         # QMessageBox.information(self, "message box", "{}{}".format(1, 2),
        #         #                         QMessageBox.Yes)
        #         QMessageBox.information(self, "message box", "input_path2 doesn't exist, please retype!",
        #                                 QMessageBox.Yes)
        #         return
        n_frame1 = videos_infos[vidx1]['nframes']
        # n_frame2 = videos_infos[vidx2]['nframes']
        if f1 < 0 or f1 >= (n_frame1-1):
            QMessageBox.information(self, "message box",
                                    "the range of frameid1 is: %d-%d, please retype!" % (0, n_frame1 - 2),
                                    QMessageBox.Yes)
            return
        if f2 <= f1 or f2 >= n_frame1:
            QMessageBox.information(self, "message box",
                                    "the range of frameid2 is: %d-%d, please retype!" % (f1+1, n_frame1 - 1),
                                    QMessageBox.Yes)
            return
        n_trackid1 = len(videos_infos[vidx1]['trackid'][f1])
        if n_trackid1 == 0:
            QMessageBox.information(self, "message box",
                                    "frameid1 %d doesn't contain any object, please retype!" % (0, f1),
                                    QMessageBox.Yes)
            return
        # n_trackid2 = len(videos_infos[vidx2]['trackid'][f2])
        # if n_trackid2 == 0:
        #     QMessageBox.information(self, "message box",
        #                             "frameid2 %d doesn't contain any object, please retype!" % (0, f2),
        #                             QMessageBox.Yes)
        #     return
        if tid1 < 0 or tid1 >= n_trackid1:
            QMessageBox.information(self, "message box",
                                    "the range of objid1 is: %d-%d, please retype!" % (0, n_trackid1 - 1),
                                    QMessageBox.Yes)
            return
        # if tid2 < 0 or tid2 >= n_trackid2:
        #     QMessageBox.information(self, "message box",
        #                             "the range of objid2 is: %d-%d, please retype!" % (0, n_trackid2 - 1),
        #                             QMessageBox.Yes)
        #     return
        path1 = videos_infos[vidx1]['img_files'][f1]
        frame1 = cv2.imread(path1)
        gt1 = videos_infos[vidx1]['gt'][f1][tid1]
        category_name1 = videos_infos[vidx1]['name'][f1][tid1]
        im_with_bb1 = draw_box_bigline(frame1, gt1, category_name1)
        im_with_bb1 = cv2.resize(im_with_bb1, (self.pic1.width(), self.pic1.height()),
                                 interpolation=cv2.INTER_CUBIC)
        im_with_bb1 = cv2.cvtColor(im_with_bb1, cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent = im_with_bb1.shape
        bytesPerLine = bytesPerComponent * width
        img1 = QtGui.QImage(im_with_bb1.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        self.pic1.setPixmap(QtGui.QPixmap.fromImage(img1).scaled(self.pic1.width(), self.pic1.height()))
        self.label_path1.setText(path1)

        self.button_track.setEnabled(False)
        self.thread_track.freshcheckBox=self.freshcheckBox.isChecked()
        self.thread_track.start()

    def call_back_track(self,path2,sia_value,category_name2,curr_bbox,is_running):
        frame2 = cv2.imread(path2)
        im_with_bb2 = draw_box_bigline(frame2, curr_bbox, category_name2)
        im_with_bb2 = cv2.resize(im_with_bb2, (self.pic2.width(), self.pic2.height()),
                                 interpolation=cv2.INTER_CUBIC)
        im_with_bb2 = cv2.cvtColor(im_with_bb2, cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent = im_with_bb2.shape
        bytesPerLine = bytesPerComponent * width
        img2 = QtGui.QImage(im_with_bb2.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        self.pic2.setPixmap(QtGui.QPixmap.fromImage(img2).scaled(self.pic2.width(), self.pic2.height()))
        self.label4.setText(str(sia_value))
        self.label_path2.setText(path2)
        QApplication.processEvents()


        self.thread_track.freshcheckBox=self.freshcheckBox.isChecked()

        if not is_running:
            self.button_track.setEnabled(True)
            self.thread_track.stop()



        # t_aera1, _, _ = self.transform3(frame1, gt1)
        # curr_bbox=gt1
        # if self.freshcheckBox.isChecked():
        #     for fi2 in range(f1+1,f2+1):
        #         path2 = videos_infos[vidx1]['img_files'][fi2]
        #         frame2 = cv2.imread(path2)
        #         curr_bboxs, curr_scores=self.adnet_inference(frame2, curr_bbox)
        #         curr_bbox=curr_bboxs[-1]
        #         self.label_path2.setText(path2)
        #         for ti in range(len(curr_scores)):
        #             t_aera2, _, _ = self.transform3(frame2, curr_bboxs[ti])
        #             output1, output2 = self.siamesenet(Variable(t_aera1).cuda(), Variable(t_aera2).cuda())
        #             euclidean_distance = F.pairwise_distance(output1, output2)
        #             sia_value = round(euclidean_distance.item(), 2)
        #             category_name2 = "step: %d/%d, score: %.2f" % (ti,len(curr_scores)-1,curr_scores[ti])
        #             im_with_bb2 = draw_box_bigline(frame2, curr_bboxs[ti], category_name2)
        #             im_with_bb2 = cv2.resize(im_with_bb2, (self.pic2.width(), self.pic2.height()),
        #                                      interpolation=cv2.INTER_CUBIC)
        #             im_with_bb2 = cv2.cvtColor(im_with_bb2, cv2.COLOR_BGR2RGB)
        #             height, width, bytesPerComponent = im_with_bb2.shape
        #             bytesPerLine = bytesPerComponent * width
        #             img2 = QtGui.QImage(im_with_bb2.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        #             self.pic2.setPixmap(QtGui.QPixmap.fromImage(img2).scaled(self.pic2.width(), self.pic2.height()))
        #             self.label4.setText(str(sia_value))
        #             QApplication.processEvents()
        #             # time.sleep(0.5)
        #             if not self.freshcheckBox.isChecked():
        #                 return
        #         # time.sleep(1)
        # else:
        #     for fi2 in range(f1+1,f2+1):
        #         path2 = videos_infos[vidx1]['img_files'][fi2]
        #         frame2 = cv2.imread(path2)
        #         curr_bboxs, curr_scores=self.adnet_inference(frame2, curr_bbox)
        #         curr_bbox=curr_bboxs[-1]
        #     t_aera2, _, _ = self.transform3(frame2, curr_bbox)
        #     output1, output2 = self.siamesenet(Variable(t_aera1).cuda(), Variable(t_aera2).cuda())
        #     euclidean_distance = F.pairwise_distance(output1, output2)
        #     sia_value = round(euclidean_distance.item(), 2)
        #     category_name2 = "score: %.2f"%(curr_scores[-1])
        #     im_with_bb2 = draw_box_bigline(frame2, curr_bbox, category_name2)
        #     im_with_bb2 = cv2.resize(im_with_bb2, (self.pic2.width(), self.pic2.height()),
        #                              interpolation=cv2.INTER_CUBIC)
        #     im_with_bb2 = cv2.cvtColor(im_with_bb2, cv2.COLOR_BGR2RGB)
        #     height, width, bytesPerComponent = im_with_bb2.shape
        #     bytesPerLine = bytesPerComponent * width
        #     img2 = QtGui.QImage(im_with_bb2.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        #     self.pic2.setPixmap(QtGui.QPixmap.fromImage(img2).scaled(self.pic2.width(), self.pic2.height()))
        #     self.label4.setText(str(sia_value))
        #     self.label_path2.setText(path2)

    

    def custom_siam(self):
        p1=self.input_path1.text()
        p2=self.input_path2.text()
        f1=self.frameid1.text()
        f2=self.frameid2.text()
        tid1=self.trackid1.text()
        tid2=self.trackid2.text()

        isint = '^-?[0-9]\d*$'
        rr1 = re.compile(isint)
        if rr1.match(f1) is None:
            QMessageBox.information(self, "message box", "Please enter an integer in the 'frameid1'",
                                    QMessageBox.Yes)
            return
        if rr1.match(f2) is None:
            QMessageBox.information(self, "message box", "Please enter an integer in the 'frameid2'",
                                    QMessageBox.Yes)
            return
        if rr1.match(tid1) is None:
            QMessageBox.information(self, "message box", "Please enter an integer in the 'objid1'",
                                    QMessageBox.Yes)
            return
        if rr1.match(tid2) is None:
            QMessageBox.information(self, "message box", "Please enter an integer in the 'objid2'",
                                    QMessageBox.Yes)
            return

        f1=int(f1)
        f2=int(f2)
        tid1=int(tid1)
        tid2=int(tid2)
        vlen = len(self.videos_infos)
        videos_infos = self.videos_infos
        vidx1=-1
        vidx2 = -1

        for i in range(vlen):
            if videos_infos[i]['img_files'][0][-35:-12]==p1:
                vidx1=i
                break
        if vidx1==-1:
            QMessageBox.information(self, "message box", "input_path1 doesn't exist, please retype!",
                                    QMessageBox.Yes)
            return
        if p1==p2:
            vidx2=vidx1
        else:
            for i in range(vlen):
                if videos_infos[i]['img_files'][0][-35:-12] == p2:
                    vidx2 = i
                    break
            if vidx2==-1:
                # QMessageBox.information(self, "message box", "{}{}".format(1, 2),
                #                         QMessageBox.Yes)
                QMessageBox.information(self, "message box", "input_path2 doesn't exist, please retype!",
                                        QMessageBox.Yes)
                return
        n_frame1=videos_infos[vidx1]['nframes']
        n_frame2 = videos_infos[vidx2]['nframes']
        if f1<0 or f1>=n_frame1:
            QMessageBox.information(self, "message box", "the range of frameid1 is: %d-%d, please retype!"%(0,n_frame1-1),
                                    QMessageBox.Yes)
            return
        if f2<0 or f2>=n_frame2:
            QMessageBox.information(self, "message box",
                                    "the range of frameid2 is: %d-%d, please retype!" % (0, n_frame2 - 1),
                                    QMessageBox.Yes)
            return
        n_trackid1 = len(videos_infos[vidx1]['trackid'][f1])
        if n_trackid1==0:
            QMessageBox.information(self, "message box",
                                    "frameid1 %d doesn't contain any object, please retype!" % (0, f1),
                                    QMessageBox.Yes)
            return
        n_trackid2 = len(videos_infos[vidx2]['trackid'][f2])
        if n_trackid2==0:
            QMessageBox.information(self, "message box",
                                    "frameid2 %d doesn't contain any object, please retype!" % (0, f2),
                                    QMessageBox.Yes)
            return
        if tid1<0 or tid1>=n_trackid1:
            QMessageBox.information(self, "message box",
                                    "the range of objid1 is: %d-%d, please retype!" % (0, n_trackid1 - 1),
                                    QMessageBox.Yes)
            return
        if tid2<0 or tid2>=n_trackid2:
            QMessageBox.information(self, "message box",
                                    "the range of objid2 is: %d-%d, please retype!" % (0, n_trackid2 - 1),
                                    QMessageBox.Yes)
            return

        path1 = videos_infos[vidx1]['img_files'][f1]
        frame1 = cv2.imread(path1)
        gt1 = videos_infos[vidx1]['gt'][f1][tid1]
        t_aera1, _, _ = self.transform3(frame1, gt1)
        path2 = videos_infos[vidx2]['img_files'][f2]
        frame2 = cv2.imread(path2)
        gt2 = videos_infos[vidx2]['gt'][f2][tid2]
        t_aera2, _, _ = self.transform3(frame2, gt2)

        output1, output2 = self.siamesenet(Variable(t_aera1).cuda(), Variable(t_aera2).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)

        sia_value=round(euclidean_distance.item(),2)

        category_name1 = videos_infos[vidx1]['name'][f1][tid1]
        category_name2 = videos_infos[vidx1]['name'][f2][tid2]

        im_with_bb1 = draw_box_bigline(frame1, gt1,category_name1)
        im_with_bb1=cv2.resize(im_with_bb1,(self.pic1.width(), self.pic1.height()), interpolation=cv2.INTER_CUBIC)
        im_with_bb1=cv2.cvtColor(im_with_bb1,cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent= im_with_bb1.shape
        bytesPerLine = bytesPerComponent* width
        img1 = QtGui.QImage(im_with_bb1.data, width, height, bytesPerLine,QtGui.QImage.Format_RGB888)
        self.pic1.setPixmap(QtGui.QPixmap.fromImage(img1).scaled(self.pic1.width(), self.pic1.height()))

        im_with_bb2 = draw_box_bigline(frame2, gt2,category_name2)
        im_with_bb2=cv2.resize(im_with_bb2,(self.pic2.width(), self.pic2.height()), interpolation=cv2.INTER_CUBIC)
        im_with_bb2=cv2.cvtColor(im_with_bb2,cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent= im_with_bb2.shape
        bytesPerLine = bytesPerComponent* width
        img2 = QtGui.QImage(im_with_bb2.data, width, height, bytesPerLine,QtGui.QImage.Format_RGB888)
        self.pic2.setPixmap(QtGui.QPixmap.fromImage(img2).scaled(self.pic2.width(), self.pic2.height()))

        self.label4.setText(str(sia_value))
        self.label_path1.setText(path1)
        self.label_path2.setText(path2)
        
    def rand_pos(self):
        p1 = "1.jpg"
        p2 = "2.jpg"
        sia_value = 0

        vlen = len(self.videos_infos)
        videos_infos=self.videos_infos
        vidx1 = random.randint(0, vlen - 1)
        while True:
            fidx1 = random.randint(0, videos_infos[vidx1]['nframes'] - 1)
            n_obj=len(videos_infos[vidx1]['trackid'][fidx1])
            if n_obj==0:
                continue
            p1 = videos_infos[vidx1]['img_files'][fidx1]
            frame1 = cv2.imread(p1)
            tid1=random.randint(0, n_obj - 1)
            gt1 = videos_infos[vidx1]['gt'][fidx1][tid1]
            t_aera1, _, _ = self.transform3(frame1, gt1)
            trackid1 = videos_infos[vidx1]['trackid'][fidx1][tid1]
            break

        # while True:
        #     vidx2 = random.randint(0, vlen - 1)
        #     if vidx2 != vidx1:
        #         break

        letf_bd = fidx1 - 20
        right_bd = fidx1 + 20
        if letf_bd < 0:
            letf_bd = 0
        if right_bd > (videos_infos[vidx1]['nframes'] - 2):
            right_bd=videos_infos[vidx1]['nframes'] - 1
        while True:
            # fidx2 = random.randint(0, videos_infos[vidx1]['nframes'] - 1)
            fidx2 = random.randint(letf_bd, right_bd)
            n_obj=len(videos_infos[vidx1]['trackid'][fidx2])
            if n_obj==0:
                continue
            p2 = videos_infos[vidx1]['img_files'][fidx2]
            frame2 = cv2.imread(p2)
            tid2=random.randint(0, n_obj - 1)
            gt2 = videos_infos[vidx1]['gt'][fidx2][tid2]
            trackid2 = videos_infos[vidx1]['trackid'][fidx2][tid2]
            if trackid1==trackid2:
                break
        t_aera2, _, _ = self.transform3(frame2, gt2)

        output1, output2 = self.siamesenet(Variable(t_aera1).cuda(), Variable(t_aera2).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)

        sia_value=round(euclidean_distance.item(),2)

        category_name1 = videos_infos[vidx1]['name'][fidx1][tid1]
        category_name2 = videos_infos[vidx1]['name'][fidx2][tid2]

        im_with_bb1 = draw_box_bigline(frame1, gt1,category_name1)
        im_with_bb1=cv2.resize(im_with_bb1,(self.pic1.width(), self.pic1.height()), interpolation=cv2.INTER_CUBIC)
        im_with_bb1=cv2.cvtColor(im_with_bb1,cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent= im_with_bb1.shape
        bytesPerLine = bytesPerComponent* width
        img1 = QtGui.QImage(im_with_bb1.data, width, height, bytesPerLine,QtGui.QImage.Format_RGB888)
        self.pic1.setPixmap(QtGui.QPixmap.fromImage(img1).scaled(self.pic1.width(), self.pic1.height()))

        # img1 = QtGui.QPixmap(p1).scaled(self.pic1.width(), self.pic1.height())
        # self.pic1.setPixmap(img1)

        im_with_bb2 = draw_box_bigline(frame2, gt2,category_name2)
        im_with_bb2=cv2.resize(im_with_bb2,(self.pic2.width(), self.pic2.height()), interpolation=cv2.INTER_CUBIC)
        im_with_bb2=cv2.cvtColor(im_with_bb2,cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent= im_with_bb2.shape
        bytesPerLine = bytesPerComponent* width
        img2 = QtGui.QImage(im_with_bb2.data, width, height, bytesPerLine,QtGui.QImage.Format_RGB888)
        self.pic2.setPixmap(QtGui.QPixmap.fromImage(img2).scaled(self.pic2.width(), self.pic2.height()))

        # img2 = QtGui.QPixmap(p2).scaled(self.pic2.width(), self.pic2.height())
        # self.pic2.setPixmap(img2)

        self.label4.setText(str(sia_value))
        self.label_path1.setText(p1)
        self.label_path2.setText(p2)

    def rand_neg(self):
        p1 = "1.jpg"
        p2 = "2.jpg"
        sia_value = 0

        vlen = len(self.videos_infos)
        videos_infos = self.videos_infos
        vidx1 = random.randint(0, vlen - 1)
        while True:
            fidx1 = random.randint(0, videos_infos[vidx1]['nframes'] - 1)
            n_obj=len(videos_infos[vidx1]['trackid'][fidx1])
            if n_obj==0:
                continue
            p1 = videos_infos[vidx1]['img_files'][fidx1]
            frame1 = cv2.imread(p1)
            tid1=random.randint(0, n_obj - 1)
            gt1 = videos_infos[vidx1]['gt'][fidx1][tid1]
            t_aera1, _, _ = self.transform3(frame1, gt1)
            trackid1 = videos_infos[vidx1]['trackid'][fidx1][tid1]
            break

        while True:
            found=False
            vidx2 = random.randint(0, vlen - 1)
            if vidx1==vidx2:
                cnt=0
                while True:
                    cnt+=1
                    fidx2 = random.randint(0, videos_infos[vidx2]['nframes'] - 1)
                    p2 = videos_infos[vidx2]['img_files'][fidx2]
                    n_obj=len(videos_infos[vidx2]['trackid'][fidx2])
                    if n_obj==0:
                        break
                    tid2=random.randint(0, n_obj - 1)
                    gt2 = videos_infos[vidx2]['gt'][fidx2][tid2]
                    trackid2 = videos_infos[vidx2]['trackid'][fidx2][tid2]
                    if trackid1 != trackid2:
                        found=True
                        break
                    elif cnt>=20:
                        break
                    else:
                        pass
            else:
                fidx2 = random.randint(0, videos_infos[vidx2]['nframes'] - 1)
                n_obj=len(videos_infos[vidx2]['trackid'][fidx2])
                if n_obj==0:
                    continue
                p2 = videos_infos[vidx2]['img_files'][fidx2]
                # frame2 = cv2.imread(p2)
                tid2=random.randint(0, n_obj - 1)
                gt2 = videos_infos[vidx2]['gt'][fidx2][tid2]
                found = True
            if found==True:
                break

        frame2 = cv2.imread(p2)
        t_aera2, _, _ = self.transform3(frame2, gt2)

        output1, output2 = self.siamesenet(Variable(t_aera1).cuda(), Variable(t_aera2).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)

        sia_value = round(euclidean_distance.item(), 2)

        category_name1 = videos_infos[vidx1]['name'][fidx1][tid1]
        category_name2 = videos_infos[vidx2]['name'][fidx2][tid2]

        im_with_bb1 = draw_box_bigline(frame1, gt1,category_name1)
        im_with_bb1=cv2.resize(im_with_bb1,(self.pic1.width(), self.pic1.height()), interpolation=cv2.INTER_CUBIC)
        im_with_bb1=cv2.cvtColor(im_with_bb1,cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent= im_with_bb1.shape
        bytesPerLine = bytesPerComponent* width
        img1 = QtGui.QImage(im_with_bb1.data, width, height, bytesPerLine,QtGui.QImage.Format_RGB888)
        self.pic1.setPixmap(QtGui.QPixmap.fromImage(img1).scaled(self.pic1.width(), self.pic1.height()))

        # img1 = QtGui.QPixmap(p1).scaled(self.pic1.width(), self.pic1.height())
        # self.pic1.setPixmap(img1)

        im_with_bb2 = draw_box_bigline(frame2, gt2,category_name2)
        im_with_bb2=cv2.resize(im_with_bb2,(self.pic2.width(), self.pic2.height()), interpolation=cv2.INTER_CUBIC)
        im_with_bb2=cv2.cvtColor(im_with_bb2,cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent= im_with_bb2.shape
        bytesPerLine = bytesPerComponent* width
        img2 = QtGui.QImage(im_with_bb2.data, width, height, bytesPerLine,QtGui.QImage.Format_RGB888)
        self.pic2.setPixmap(QtGui.QPixmap.fromImage(img2).scaled(self.pic2.width(), self.pic2.height()))

        # im_with_bb1 = draw_box(frame1, gt1)
        # height, width, bytesPerComponent= frame1.shape
        # bytesPerLine = bytesPerComponent* width
        # im_with_bb1=cv2.cvtColor(im_with_bb1,cv2.COLOR_BGR2RGB)
        # img1 = QtGui.QImage(im_with_bb1.data, width, height, bytesPerLine,QtGui.QImage.Format_RGB888)
        # self.pic1.setPixmap(QtGui.QPixmap.fromImage(img1).scaled(self.pic1.width(), self.pic1.height()))

        # # img1 = QtGui.QPixmap(p1).scaled(self.pic1.width(), self.pic1.height())
        # # self.pic1.setPixmap(img1)

        # im_with_bb2 = draw_box(frame2, gt2)
        # height, width, bytesPerComponent= frame2.shape
        # bytesPerLine = bytesPerComponent* width
        # im_with_bb2=cv2.cvtColor(im_with_bb2,cv2.COLOR_BGR2RGB)
        # img2 = QtGui.QImage(im_with_bb2.data, width, height, bytesPerLine,QtGui.QImage.Format_RGB888)
        # self.pic2.setPixmap(QtGui.QPixmap.fromImage(img2).scaled(self.pic2.width(), self.pic2.height()))

        # img2 = QtGui.QPixmap(p2).scaled(self.pic2.width(), self.pic2.height())
        # self.pic2.setPixmap(img2)

        self.label4.setText(str(sia_value))
        self.label_path1.setText(p1)
        self.label_path2.setText(p2)

    def rand_neg_same_frame(self):
        p1 = "1.jpg"
        p2 = "2.jpg"
        sia_value = 0

        vlen = len(self.videos_infos)
        videos_infos = self.videos_infos
        vidx1 = random.randint(0, vlen - 1)
        while True:
            fidx1 = random.randint(0, videos_infos[vidx1]['nframes'] - 1)
            n_obj=len(videos_infos[vidx1]['trackid'][fidx1])
            if n_obj==0:
                continue
            p1 = videos_infos[vidx1]['img_files'][fidx1]
            frame1 = cv2.imread(p1)
            tid1=random.randint(0, n_obj - 1)
            gt1 = videos_infos[vidx1]['gt'][fidx1][tid1]
            t_aera1, _, _ = self.transform3(frame1, gt1)
            trackid1 = videos_infos[vidx1]['trackid'][fidx1][tid1]
            break

        opts['imgSize'] = [frame1.shape[0],frame1.shape[1]]
        pn=random.randint(0, 1)
        if pn:
            gt2 = gen_samples('gaussian', gt1, 1, opts, 2, 10)
        else:
            gt2 = gen_samples('gaussian', gt1, 1, opts, 0.1, 5)

        gt2=gt2[0]

        t_aera2, _, _ = self.transform3(frame1, gt2)

        output1, output2 = self.siamesenet(Variable(t_aera1).cuda(), Variable(t_aera2).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)

        sia_value = round(euclidean_distance.item(), 2)

        category_name1 = videos_infos[vidx1]['name'][fidx1][tid1]
        category_name2 = "random box"

        im_with_bb1 = draw_box_bigline(frame1, gt1,category_name1)
        im_with_bb1=cv2.resize(im_with_bb1,(self.pic1.width(), self.pic1.height()), interpolation=cv2.INTER_CUBIC)
        im_with_bb1=cv2.cvtColor(im_with_bb1,cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent= im_with_bb1.shape
        bytesPerLine = bytesPerComponent* width
        img1 = QtGui.QImage(im_with_bb1.data, width, height, bytesPerLine,QtGui.QImage.Format_RGB888)
        self.pic1.setPixmap(QtGui.QPixmap.fromImage(img1).scaled(self.pic1.width(), self.pic1.height()))

        # img1 = QtGui.QPixmap(p1).scaled(self.pic1.width(), self.pic1.height())
        # self.pic1.setPixmap(img1)

        im_with_bb2 = draw_box_bigline(frame1, gt2,category_name2)
        im_with_bb2=cv2.resize(im_with_bb2,(self.pic2.width(), self.pic2.height()), interpolation=cv2.INTER_CUBIC)
        im_with_bb2=cv2.cvtColor(im_with_bb2,cv2.COLOR_BGR2RGB)
        img2 = QtGui.QImage(im_with_bb2.data, width, height, bytesPerLine,QtGui.QImage.Format_RGB888)
        self.pic2.setPixmap(QtGui.QPixmap.fromImage(img2).scaled(self.pic2.width(), self.pic2.height()))

        self.label4.setText(str(sia_value))
        self.label_path1.setText(p1)
        self.label_path2.setText(p1)
