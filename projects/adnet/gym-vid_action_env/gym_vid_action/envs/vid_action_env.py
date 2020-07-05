import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

import numpy as np
import cv2
import torch

# try:
#     import hfo_py
# except ImportError as e:
#     raise error.DependencyNotInstalled("{}. (HINT: you can install HFO dependencies with 'pip install gym[soccer].)'".format(e))

import logging
logger = logging.getLogger(__name__)

class ActionEnv(gym.Env, utils.EzPickle):
    def __init__(self):
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(112),
            spaces.Discrete(112),
            spaces.Discrete(3)))
        self.seed()

    def init_data(self,videos_infos, opts, transform, do_action,overlap_ratio):
        self.videos_infos=videos_infos
        self.opts = opts
        self.transform = transform
        # self.args = args
        self.do_action = do_action
        self.overlap_ratio = overlap_ratio

        self.videos = []  # list of clips dict

        self.RL_steps = self.opts['train']['RL_steps']  # clip length

        vid_idxs = np.random.permutation(len(self.videos_infos))
        # print("num videos: %d "%len(vid_idxs))
        for vid_idx in vid_idxs:
            # dict consist of set of clips in ONE video
            clips = {
                'img_path': [],
                'frame_start': [],
                'frame_end': [],
                'init_bbox': [],
                'end_bbox': [],
                'vid_idx': [],
            }
            vid_info = self.videos_infos[vid_idx]
            if self.RL_steps is None:
                self.RL_steps = len(vid_info['gt']) - 1
                vid_clip_starts = [0]
                vid_clip_ends = [len(vid_info['gt']) - 1]
            else:
                vid_clip_starts = np.array(range(len(vid_info['gt']) - self.RL_steps))
                vid_clip_starts = np.random.permutation(vid_clip_starts)
                vid_clip_ends = vid_clip_starts + self.RL_steps

            # number of clips in one video
            num_train_clips = min(self.opts['train']['rl_num_batches'], len(vid_clip_starts))

            # print("num_train_clips of vid " + str(vid_idx) + ": ", str(num_train_clips))

            for clipIdx in range(num_train_clips):
                frameStart = vid_clip_starts[clipIdx]
                frameEnd = vid_clip_ends[clipIdx]

                clips['img_path'].append(vid_info['img_files'][frameStart:frameEnd])
                clips['frame_start'].append(frameStart)
                clips['frame_end'].append(frameEnd)
                clips['init_bbox'].append(vid_info['gt'][frameStart])
                clips['end_bbox'].append(vid_info['gt'][frameEnd])
                clips['vid_idx'].append(vid_idx)

            if num_train_clips > 0:  # small hack
                self.videos.append(clips)
        self.clip_idx = -1  # hack for reset function
        self.vid_idx = 0

        self.state = None  # current bbox
        self.gt = None  # end bbox
        self.current_img = None  # current image frame
        self.current_img_cuda = None
        self.current_patch = None  # current patch (transformed)
        self.current_patch_cuda = None
        self.current_img_idx = 0
        self.finish_epoch=False
        # self.box_history_clip=[]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,action):
        info = {
            'finish_epoch': False
        }
        if action == self.opts['stop_action']:
            reward, done,_ = self.go_to_next_frame()

            info['finish_epoch'] = self.finish_epoch

        else:   # just go to the next patch (still same frame/current_img)
            reward = 0
            done = False
            iou_bef=self.overlap_ratio(self.gt, self.state)
            # do action
            self.state = self.do_action(self.state, self.opts, action, self.current_img.shape)
            iou_aft=self.overlap_ratio(self.gt, self.state)
            iou_change=iou_aft-iou_bef
            iou_ratio=20
            if iou_change>0.01:
                reward=iou_change*iou_ratio-0.1
            elif iou_change<0.01:
                reward=iou_change*iou_ratio-0.1
            self.current_patch, _, _, _ = self.transform(self.current_img, self.state)

        return self.current_patch,self.state, reward, done, info

    def reset(self):
        while True:
            self.clip_idx += 1

            # if the clips in a video are finished... go to the next video
            if self.clip_idx >= len(self.videos[self.vid_idx]['frame_start']):
                self.vid_idx += 1
                self.clip_idx = 0
                if self.vid_idx >= len(self.videos):
                    self.vid_idx = 0
                    # one epoch finish... need to reinitialize the class to use this again randomly
                    self.finish_epoch=True
                    return None

            # initialize state, gt, current_img_idx, current_img, and current_patch with new clip
            self.state = self.videos[self.vid_idx]['init_bbox'][self.clip_idx]
            self.gt = self.videos[self.vid_idx]['end_bbox'][self.clip_idx]
            # if self.state==[0,0,0,0]:
            #     print("debug")

            # frameStart = self.videos[self.vid_idx]['frame_start'][self.clip_idx]
            #self.current_img_idx = 1  # self.current_img_idx = frameStart + 1
            self.current_img_idx = 1   #the frameStart(the 0th img,idx:0) is for initial, the current_img(idx:1) is for training.
            self.current_img = cv2.imread(self.videos[self.vid_idx]['img_path'][self.clip_idx][self.current_img_idx])
            # imgcuda = self.current_img.copy()
            imgcuda = self.current_img.astype(np.float32)
            self.current_img=torch.from_numpy(imgcuda).cuda()
            self.current_patch, _, _, _ = self.transform(self.current_img, np.array(self.state))
            #Modified by zb --- 2019-11-16 22:11:16 --- to check : at this step ,the data of patch seems have some problem\
            #because some data results are under zero

            if self.gt != '':  # small hack
                break
        return self.current_patch

    def get_current_patch(self):
        return self.current_patch

    def get_current_train_vid_idx(self):
        return self.videos[self.vid_idx]['vid_idx'][0]

    def get_state(self):
        return self.state

    def get_current_img(self):
        return self.current_img

    def go_to_next_frame(self):
        # self.box_history_clip = []
        self.current_img_idx += 1
        # finish_epoch = False

        # if already in the end of a clip...
        #aaa=self.current_img_idx
        #bbb=len(self.videos[self.vid_idx]['img_path'][self.clip_idx])
        if self.current_img_idx >= len(self.videos[self.vid_idx]['img_path'][self.clip_idx]):
            # calculate reward before reset
            reward = self.reward_original(np.array(self.gt), np.array(self.state))

            # print("reward=" + str(reward))

            # reset (reset state, gt, current_img_idx, current_img and current_img_patch)
            # self.finish_epoch,_ = self.reset()  # go to the next clip (or video)
            self.reset()

            done = True  # done means one clip is finished

        # just go to the next frame (means new patch and new image)
        else:
            reward = 0
            done = False
            # note: reset already read the current_img and current_img_patch
            self.current_img = cv2.imread(self.videos[self.vid_idx]['img_path'][self.clip_idx][self.current_img_idx])
            imgcuda = self.current_img.astype(np.float32)
            self.current_img = torch.from_numpy(imgcuda).cuda()
            self.current_patch, _, _, _ = self.transform(self.current_img, self.state)

        return reward, done,self.finish_epoch

    def reward_original(self,gt, box):
        iou = self.overlap_ratio(gt, box)
        if iou > 0.7:
            reward = 1
        else:
            reward = -1

        return reward







    # metadata = {'render.modes': ['human']}

#     def __init__(self):
#         self.viewer = None
#         self.server_process = None
#         self.server_port = None
#         self.hfo_path = hfo_py.get_hfo_path()
#         self._configure_environment()
#         self.env = hfo_py.HFOEnvironment()
#         self.env.connectToServer(config_dir=hfo_py.get_config_path())
#         self.observation_space = spaces.Box(low=-1, high=1,
#                                             shape=(self.env.getStateSize()))
#         # Action space omits the Tackle/Catch actions, which are useful on defense
#         self.action_space = spaces.Tuple((spaces.Discrete(3),
#                                           spaces.Box(low=0, high=100, shape=1),
#                                           spaces.Box(low=-180, high=180, shape=1),
#                                           spaces.Box(low=-180, high=180, shape=1),
#                                           spaces.Box(low=0, high=100, shape=1),
#                                           spaces.Box(low=-180, high=180, shape=1)))
#         self.status = hfo_py.IN_GAME

#     def __del__(self):
#         self.env.act(hfo_py.QUIT)
#         self.env.step()
#         os.kill(self.server_process.pid, signal.SIGINT)
#         if self.viewer is not None:
#             os.kill(self.viewer.pid, signal.SIGKILL)

#     def _configure_environment(self):
#         """
#         Provides a chance for subclasses to override this method and supply
#         a different server configuration. By default, we initialize one
#         offense agent against no defenders.
#         """
#         self._start_hfo_server()

#     def _start_hfo_server(self, frames_per_trial=500,
#                           untouched_time=100, offense_agents=1,
#                           defense_agents=0, offense_npcs=0,
#                           defense_npcs=0, sync_mode=True, port=6000,
#                           offense_on_ball=0, fullstate=True, seed=-1,
#                           ball_x_min=0.0, ball_x_max=0.2,
#                           verbose=False, log_game=False,
#                           log_dir="log"):
#         """
#         Starts the Half-Field-Offense server.
#         frames_per_trial: Episodes end after this many steps.
#         untouched_time: Episodes end if the ball is untouched for this many steps.
#         offense_agents: Number of user-controlled offensive players.
#         defense_agents: Number of user-controlled defenders.
#         offense_npcs: Number of offensive bots.
#         defense_npcs: Number of defense bots.
#         sync_mode: Disabling sync mode runs server in real time (SLOW!).
#         port: Port to start the server on.
#         offense_on_ball: Player to give the ball to at beginning of episode.
#         fullstate: Enable noise-free perception.
#         seed: Seed the starting positions of the players and ball.
#         ball_x_[min/max]: Initialize the ball this far downfield: [0,1]
#         verbose: Verbose server messages.
#         log_game: Enable game logging. Logs can be used for replay + visualization.
#         log_dir: Directory to place game logs (*.rcg).
#         """
#         self.server_port = port
#         cmd = self.hfo_path + \
#               " --headless --frames-per-trial %i --untouched-time %i --offense-agents %i"\
#               " --defense-agents %i --offense-npcs %i --defense-npcs %i"\
#               " --port %i --offense-on-ball %i --seed %i --ball-x-min %f"\
#               " --ball-x-max %f --log-dir %s"\
#               % (frames_per_trial, untouched_time, offense_agents,
#                  defense_agents, offense_npcs, defense_npcs, port,
#                  offense_on_ball, seed, ball_x_min, ball_x_max,
#                  log_dir)
#         if not sync_mode: cmd += " --no-sync"
#         if fullstate:     cmd += " --fullstate"
#         if verbose:       cmd += " --verbose"
#         if not log_game:  cmd += " --no-logging"
#         print('Starting server with command: %s' % cmd)
#         self.server_process = subprocess.Popen(cmd.split(' '), shell=False)
#         time.sleep(10) # Wait for server to startup before connecting a player

#     def _start_viewer(self):
#         """
#         Starts the SoccerWindow visualizer. Note the viewer may also be
#         used with a *.rcg logfile to replay a game. See details at
#         https://github.com/LARG/HFO/blob/master/doc/manual.pdf.
#         """
#         cmd = hfo_py.get_viewer_path() +\
#               " --connect --port %d" % (self.server_port)
#         self.viewer = subprocess.Popen(cmd.split(' '), shell=False)

#     def _step(self, action):
#         self._take_action(action)
#         self.status = self.env.step()
#         reward = self._get_reward()
#         ob = self.env.getState()
#         episode_over = self.status != hfo_py.IN_GAME
#         return ob, reward, episode_over, {}

#     def _take_action(self, action):
#         """ Converts the action space into an HFO action. """
#         action_type = ACTION_LOOKUP[action[0]]
#         if action_type == hfo_py.DASH:
#             self.env.act(action_type, action[1], action[2])
#         elif action_type == hfo_py.TURN:
#             self.env.act(action_type, action[3])
#         elif action_type == hfo_py.KICK:
#             self.env.act(action_type, action[4], action[5])
#         else:
#             print('Unrecognized action %d' % action_type)
#             self.env.act(hfo_py.NOOP)

#     def _get_reward(self):
#         """ Reward is given for scoring a goal. """
#         if self.status == hfo_py.GOAL:
#             return 1
#         else:
#             return 0

#     def _reset(self):
#         """ Repeats NO-OP action until a new episode begins. """
#         while self.status == hfo_py.IN_GAME:
#             self.env.act(hfo_py.NOOP)
#             self.status = self.env.step()
#         while self.status != hfo_py.IN_GAME:
#             self.env.act(hfo_py.NOOP)
#             self.status = self.env.step()
#         return self.env.getState()

#     def _render(self, mode='human', close=False):
#         """ Viewer only supports human mode currently. """
#         if close:
#             if self.viewer is not None:
#                 os.kill(self.viewer.pid, signal.SIGKILL)
#         else:
#             if self.viewer is None:
#                 self._start_viewer()

# ACTION_LOOKUP = {
#     0 : hfo_py.DASH,
#     1 : hfo_py.TURN,
#     2 : hfo_py.KICK,
#     3 : hfo_py.TACKLE, # Used on defense to slide tackle the ball
#     4 : hfo_py.CATCH,  # Used only by goalie to catch the ball
# }
