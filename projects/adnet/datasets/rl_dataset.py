# pytorch dataset for SL learning
# matlab code (line 26-33):
# https://github.com/hellbell/ADNet/blob/master/train/adnet_train_SL.m
# reference:
# https://github.com/amdegroot/ssd.pytorch/blob/master/data/voc0712.py
import multiprocessing
import os
import copy
import cv2
import numpy as np
import torch
import torch.utils.data as data
from datasets.get_train_dbs import get_train_dbs
from utils.get_video_infos import get_video_infos

import time
from trainers.RL_tools import TrackingEnvironment
from utils.augmentations import ADNet_Augmentation,ADNet_Augmentation2
from utils.display import display_result, draw_box
from torch.distributions import Categorical
from utils.my_util import get_ILSVRC_videos_infos

class RLDataset(data.Dataset):

    def __init__(self, net, domain_specific_nets, train_videos, opts, args):
        # self.env = None

        # these lists won't include the ground truth
        self.action_list = []  # a_t,l  # argmax of self.action_prob_list
        self.action_prob_list = []  # output of network (fc6_out)
        self.log_probs_list = []  # log probs from each self.action_prob_list member
        self.reward_list = []  # tracking score
        self.patch_list = []  # input of network
        self.action_dynamic_list = []  # action_dynamic used for inference (means before updating the action_dynamic)
        self.result_box_list = []
        self.vid_idx_list = []
        self.opts=opts
        self.args=args
        self.reset(net, domain_specific_nets, train_videos, opts, args)

    def __getitem__(self, index):

        # return self.action_list[index], self.action_prob_list[index], self.log_probs_list[index], \
        #        self.reward_list[index], self.patch_list[index], self.action_dynamic_list[index], \
        #        self.result_box_list[index]

        # TODO: currently only log_probs_list, reward_list, and vid_idx_list contains data
        return self.log_probs_list[index], self.reward_list[index], self.vid_idx_list[index]

    def __len__(self):
        return len(self.log_probs_list)

    def gen_data(self,videos_infos,transform,net,t_action_list,t_log_probs_list,t_reward_list,t_action_prob_list,t_patch_list,t_action_dynamic_list,t_result_box_list,t_vid_idx_list,lock):
        # net=copy.deepcopy(net1)
        env = TrackingEnvironment(videos_infos, self.opts, transform=transform, args=self.args)
        clip_idx = 0
        tic = time.time()
        # n_pos_clip = 0
        # n_neg_clip = 0
        action_list_ = []  # a_t,l  # argmax of self.action_prob_list
        log_probs_list_ = []  # log probs from each self.action_prob_list member
        reward_list_ = []  # tracking score
        # action_prob_list_ = []  # output of network (fc6_out)
        # patch_list_ = []  # input of network
        # action_dynamic_list_ = []  # action_dynamic used for inference (means before updating the action_dynamic)
        # result_box_list_ = []
        vid_idx_list_ = []
        while True:  # for every clip (l)
            num_step_history = []  # T_l
            num_frame = 1  # the first frame won't be tracked..
            t = 0
            box_history_clip = []  # for checking oscillation in a clip

            if self.args.cuda:
                net.module.reset_action_dynamic()
            else:
                net.reset_action_dynamic()  # action dynamic should be in a clip (what makes sense...)

            while True:  # for every frame in a clip (t)
                # tic = time.time()

                if self.args.display_images:
                    im_with_bb = display_result(env.get_current_img(), env.get_state())
                    # tem=self.env.get_current_patch_unprocessed()
                    cv2.imshow('patch', env.get_current_patch_unprocessed())
                    cv2.waitKey(1)
                else:
                    im_with_bb = draw_box(env.get_current_img(), env.get_state())

                if self.args.save_result_images:
                    if not os.path.exists('images'):
                        os.makedirs('images')
                    cv2.imwrite('images/' + str(clip_idx) + '-' + str(t) + '.jpg', im_with_bb)

                curr_patch = env.get_current_patch()
                # if self.args.cuda:
                #     curr_patch = curr_patch.cuda()

                # patch_list_.append(curr_patch.cpu().data.numpy())  # TODO: saving patch takes cuda memory

                # TODO: saving action_dynamic takes cuda memory
                # if args.cuda:
                #     action_dynamic_list_.append(net.module.get_action_dynamic())
                # else:
                #     action_dynamic_list_.append(net.get_action_dynamic())

                # curr_patch = curr_patch.unsqueeze(0)  # 1 batch input [1, curr_patch.shape]

                # load ADNetDomainSpecific with video index
                if self.args.multidomain:
                    vid_idx = env.get_current_train_vid_idx()
                else:
                    vid_idx = 0
                # if self.args.cuda:
                #     self.net.module.load_domain_specific(domain_specific_nets[vid_idx])
                # else:
                #     self.net.load_domain_specific(domain_specific_nets[vid_idx])

                fc6_out, fc7_out = net.forward(curr_patch, update_action_dynamic=True)

                if self.args.cuda:
                    action = np.argmax(fc6_out.detach().cpu().numpy())  # TODO: really okay to detach?
                    action_prob = fc6_out.detach().cpu().numpy()[0][action]
                else:
                    action = np.argmax(fc6_out.detach().numpy())  # TODO: really okay to detach?
                    action_prob = fc6_out.detach().numpy()[0][action]

                m = Categorical(probs=fc6_out)
                action_ = m.sample()  # action and action_ are same value. Only differ in the type (int and tensor)

                log_probs_list_.append(m.log_prob(action_).cpu().data.numpy())
                vid_idx_list_.append(vid_idx)
                action_list_.append(action)
                # TODO: saving action_prob_list takes cuda memory
                # action_prob_list_.append(action_prob)

                # if action==opts['stop_action']:
                #    print(action)

                new_state, reward, done, info = env.step(action)

                # check oscillating
                # if any((np.array(new_state).round() == x).all() for x in np.array(box_history_clip).round()):
                if ((action != self.opts['stop_action']) and any(
                        (np.array(new_state).round() == x).all() for x in np.array(box_history_clip).round())):
                    action = self.opts['stop_action']
                    reward, done, finish_epoch = env.go_to_next_frame()
                    info['finish_epoch'] = finish_epoch

                # check if number of action is already too much
                if t > self.opts['num_action_step_max']:
                    action = self.opts['stop_action']
                    reward, done, finish_epoch = env.go_to_next_frame()
                    info['finish_epoch'] = finish_epoch

                # TODO: saving result_box takes cuda memory
                # result_box_list_.append(list(new_state))
                box_history_clip.append(list(new_state))

                t += 1

                if action == self.opts['stop_action']:
                    num_frame += 1
                    num_step_history.append(t)
                    t = 0

                # toc = time.time() - tic
                # print('forward time (clip ' + str(clip_idx) + " - frame " + str(num_frame) + " - t " + str(t) + ") = "
                #       + str(toc) + " s")

                if done:  # if finish the clip
                    break

            tracking_scores_size = np.array(num_step_history).sum()
            tracking_scores = np.full(tracking_scores_size, reward)  # seems no discount factor whatsoever

            reward_list_.extend(tracking_scores)
            # self.reward_list.append(tracking_scores)

            clip_idx += 1
            # if reward > 0:
            #     n_pos_clip += 1
            # else:
            #     n_neg_clip += 1

            if clip_idx % 1000 == 0 or info['finish_epoch']:
                toc = time.time() - tic
                print('forward time (clip ' + str(clip_idx) + ") = "
                      + str(toc) + " s")
                # print('n_pos_clip: ' + str(n_pos_clip) + ' all clip: ' + str(n_pos_clip + n_neg_clip)
                #       + ' ; n_pos_data: ' + str(self.reward_list.count(1)) + ' n_all_data: ' + str(
                #     len(self.reward_list)))
                tic = time.time()

            if info['finish_epoch']:
                break
        try:
            lock.acquire()
            t_action_list.extend(action_list_)
            t_log_probs_list.extend(log_probs_list_)
            t_reward_list.extend(reward_list_)
            # t_action_prob_list.extend(action_prob_list_)
            # t_patch_list.extend(patch_list_)
            # t_action_dynamic_list.extend(action_dynamic_list_)
            # t_result_box_list.extend(result_box_list_)
            t_vid_idx_list.extend(vid_idx_list_)
        except Exception as err:
            raise err
        finally:
            lock.release()

    def reset(self, net, domain_specific_nets, train_videos, opts, args):
        self.action_list = []  # a_t,l  # argmax of self.action_prob_list
        self.action_prob_list = []  # output of network (fc6_out)
        self.log_probs_list = []  # log probs from each self.action_prob_list member
        self.reward_list = []  # tracking score
        self.patch_list = []  # input of network
        self.action_dynamic_list = []  # action_dynamic used for inference (means before updating the action_dynamic)
        self.result_box_list = []
        self.vid_idx_list = []

        print('generating reinforcement learning dataset')
        # transform = ADNet_Augmentation(opts)
        mean = np.array(opts['means'], dtype=np.float32)
        mean = torch.from_numpy(mean).cuda()
        transform = ADNet_Augmentation2(opts,mean)

        if train_videos==None:
            t_action_list = multiprocessing.Manager().list()
            t_log_probs_list = multiprocessing.Manager().list()
            t_reward_list = multiprocessing.Manager().list()
            t_action_prob_list = multiprocessing.Manager().list()
            t_patch_list = multiprocessing.Manager().list()
            t_action_dynamic_list = multiprocessing.Manager().list()
            t_result_box_list = multiprocessing.Manager().list()
            t_vid_idx_list = multiprocessing.Manager().list()
            videos_infos,_=get_ILSVRC_videos_infos()
            print("num all videos: %d " % len(videos_infos))
            # cpu_num = 27
            cpu_num = 4
            all_vid_num = len(videos_infos)
            if all_vid_num < cpu_num:
                cpu_num = all_vid_num
            every_gpu_vid = all_vid_num // cpu_num
            vid_paths_as = []
            for gn in range(cpu_num - 1):
                vid_paths_as.append(videos_infos[gn * every_gpu_vid:(gn + 1) * every_gpu_vid])
            vid_paths_as.append(videos_infos[(cpu_num - 1) * every_gpu_vid:])

            lock = multiprocessing.Manager().Lock()
            record = []
            print('before process', end=' : ')
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            for i in range(cpu_num):
                process = multiprocessing.Process(target=self.gen_data,
                                                  args=(vid_paths_as[i], transform, net,t_action_list,t_log_probs_list,t_reward_list,t_action_prob_list,t_patch_list,t_action_dynamic_list,t_result_box_list,t_vid_idx_list, lock))
                process.start()
                record.append(process)
            for process in record:
                process.join()
            print('after process', end=' : ')
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            self.action_list = list(t_action_list)
            self.action_prob_list = list(t_action_prob_list)
            self.log_probs_list = list(t_log_probs_list)
            self.reward_list = list(t_reward_list)
            self.patch_list = list(t_patch_list)
            self.action_dynamic_list = list(t_action_dynamic_list)
            self.result_box_list = list(t_result_box_list)
            self.vid_idx_list = list(t_vid_idx_list)
        else:
            #not implement yet
            pass
        print('generating reinforcement learning dataset finish')

