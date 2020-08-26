import _init_paths
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from torch.distributions import kl_divergence
import torch.utils.data as data
import copy
import cv2
import os
import time
import argparse
import torch.optim as optim
from models.vggm import vggm
from tensorboardX import SummaryWriter
from utils.get_train_videos import get_train_videos
from datasets.sl_dataset import initialize_pos_neg_dataset
from utils.augmentations import ADNet_Augmentation,ADNet_Augmentation2
import torch.backends.cudnn as cudnn
from options.general2 import opts
from prefetch_generator import BackgroundGenerator
from utils.do_action import do_action
from utils.my_util import get_ILSVRC_eval_infos,cal_iou

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='ADNet training')

# parser.add_argument('--resume', default='weights/ADNet_SL_backup.pth', type=str, help='Resume from checkpoint')
# parser.add_argument('--resume', default='weights/ADNet_RL_2epoch8_backup.pth', type=str, help='Resume from checkpoint')
# parser.add_argument('--resume', default='weights/ADNet_SL_epoch27_final.pth', type=str, help='Resume from checkpoint')
parser.add_argument('--resume_actor', default='', type=str, help='Actor Resume from checkpoint')
parser.add_argument('--resume_critic', default='', type=str, help='Critic Resume from checkpoint')
parser.add_argument('--num_workers', default=6, type=int, help='Number of workers used in dataloading')
parser.add_argument('--start_iter', default=2, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--visualize', default=True, type=str2bool, help='Use tensorboardx to for loss visualization')
parser.add_argument('--send_images_to_visualization', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default='weights_del', type=str, help='Location to save checkpoint models')
parser.add_argument('--tensorlogdir', default='tensorboardx_log_del', type=str, help='Location to save tensorboardx_log')
parser.add_argument('--train_consecutive', default=False, type=str2bool, help='Whether to train consecutive frames')
parser.add_argument('--train_mul_step', default=False, type=str2bool, help='Whether to train multiple steps')

parser.add_argument('--save_file_critic', default='ADNet_CRITIC_', type=str, help='save file part of file name for CRITIC')
parser.add_argument('--save_file_actor', default='ADNet_ACTOR_', type=str, help='save file part of file name for ACTOR')
parser.add_argument('--start_epoch', default=0, type=int, help='Begin counting epochs starting from this value')

parser.add_argument('--run_supervised', default=True, type=str2bool, help='Whether to run supervised learning or not')
parser.add_argument('--run_ppo', default=True, type=str2bool, help='Whether to run ppo learning or not')
parser.add_argument('--multidomain', default=False, type=str2bool, help='Separating weight for each videos (default) or not')

parser.add_argument('--save_result_images', default=False, type=str2bool, help='Whether to save the results or not. Save folder: images/')
parser.add_argument('--display_images', default=False, type=str2bool, help='Whether to display images or not')


#torch.set_default_tensor_type('torch.cuda.FloatTensor')
#Train('vggm')
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=1.0),   # KL penalty; lam is actually beta from the PPO paper
    dict(name='clip', epsilon=0.1),           # Clipped surrogate objective, find this is better
][1]
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
GAMMA=0.95

class Actor(nn.Module):
    def __init__(self, base_network, opts, num_classes=11, phase='pi', num_history=10, use_gpu=True):
        super(Actor, self).__init__()
        self.num_classes = num_classes
        self.num_history=num_history
        self.phase = phase
        self.opts = opts
        self.use_gpu = use_gpu
        self.base_network = base_network
        self.action_history = np.full(self.num_history, -1)

        self.action_dynamic_size = self.num_classes * self.num_history
        self.action_dynamic = torch.Tensor(np.zeros(self.action_dynamic_size))

        # if self.use_gpu:
            # self.action_dynamic = self.action_dynamic.cuda()
        self._build_anet(self.phase)

    def _build_anet(self,phase):
        self.fc4_5 = nn.Sequential(
            nn.Linear(4608, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),  # [3]
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.mu = nn.Linear(512 + self.action_dynamic_size, self.num_classes+2)

        self.sigma= nn.Linear(512 + self.action_dynamic_size, self.num_classes+2)

    def forward(self, x,action_dynamic=None, update_action_dynamic=False):
        assert x is not None
        x = self.base_network(x)
        x = x.view(x.size(0), -1)
        x = self.fc4_5(x)

        if action_dynamic is None:
            x = torch.cat((x, self.action_dynamic.expand(x.shape[0], self.action_dynamic.shape[0])), 1)
        else:
            x = torch.cat((x, action_dynamic))

        small=1e-6
        mu = torch.clamp(self.mu(x),small,2-small)
        sigma = torch.clamp(self.sigma(x),small,2-small)

        prob_out=Normal(mu,sigma)
        action_out=prob_out.sample()
        if update_action_dynamic:
            selected_action = np.argmax(action_out.detach().cpu().numpy()[0:11])  # TODO: really okay to detach?
            self.action_history[1:] = self.action_history[0:-1]
            self.action_history[0] = selected_action
            self.update_action_dynamic(self.action_history)

        return prob_out,action_out

class Critic(nn.Module):
    def __init__(self, base_network, opts, num_classes=11, phase='pi', num_history=10, use_gpu=True):
        super(Critic, self).__init__()
        self.num_classes = num_classes
        self.num_history=num_history
        self.phase = phase
        self.opts = opts
        self.use_gpu = use_gpu
        self.base_network = base_network
        self.action_history = np.full(self.num_history, -1)

        self.action_dynamic_size = self.num_classes * self.num_history
        self.action_dynamic = torch.Tensor(np.zeros(self.action_dynamic_size))

        # if self.use_gpu:
        #     self.action_dynamic = self.action_dynamic.cuda()
        self._build_anet(self.phase)

    def _build_anet(self,phase):
        self.fc4_5 = nn.Sequential(
            nn.Linear(4608, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),  # [3]
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.value = nn.Linear(512 + self.action_dynamic_size, 1)

    def forward(self, x,action_dynamic=None, update_action_dynamic=False):
        assert x is not None
        x = self.base_network(x)
        x = x.view(x.size(0), -1)
        x = self.fc4_5(x)
        if action_dynamic is None:
            x = torch.cat((x, self.action_dynamic.expand(x.shape[0], self.action_dynamic.shape[0])), 1)
        else:
            x = torch.cat((x, action_dynamic))

        value=self.value(x)
        if update_action_dynamic:
            selected_action = np.argmax(value.detach().cpu().numpy()[0:11])  # TODO: really okay to detach?
            self.action_history[1:] = self.action_history[0:-1]
            self.action_history[0] = selected_action
            self.update_action_dynamic(self.action_history)
        
        return value

class PPO(object):
    def __init__(self, base_network, opts,resume_actor=None,resume_critic=None, num_classes=11, phase='pi', num_history=10, use_gpu=True):
        if resume_actor and resume_critic:
            self.critic= torch.load(resume_critic)
            self.main_actor=torch.load(resume_actor)
        else:
            self.main_actor=Actor(base_network, opts)
            self.critic=Critic(base_network, opts)
            self.main_actor=self.actor_init(self.main_actor)
            self.critic = self.critic_init(self.critic)
        self.target_actor=copy.deepcopy(self.main_actor)
        self.main_actor.eval()
        self.target_actor.eval()
        self.critic.eval()
        self.A_UPDATE_STEPS=A_UPDATE_STEPS
        self.C_UPDATE_STEPS=C_UPDATE_STEPS
        self.criterion = nn.MSELoss(reduce=False, size_average=False)
        self.buffer_s, self.buffer_a, self.buffer_r = [], [], []
        self.actor_optimizer = optim.SGD( [
            {'params': self.main_actor.base_network.parameters(), 'lr': 1e-4},
            {'params': self.main_actor.fc4_5.parameters()},
            {'params': self.main_actor.mu.parameters()},
            {'params': self.main_actor.sigma.parameters()}],
            lr=1e-3, momentum=opts['train']['momentum'], weight_decay=opts['train']['weightDecay'])    
        self.critic_optimizer = optim.SGD( [
            {'params': self.critic.base_network.parameters(), 'lr': 1e-4},
            {'params': self.critic.fc4_5.parameters()},
            {'params': self.critic.value.parameters()}],
            lr=1e-3, momentum=opts['train']['momentum'], weight_decay=opts['train']['weightDecay'])

    def actor_init(self,net):
        scal = torch.Tensor([0.01])
        # fc 4
        nn.init.normal_(net.fc4_5[0].weight.data)
        net.fc4_5[0].weight.data = net.fc4_5[0].weight.data * scal.expand_as(net.fc4_5[0].weight.data)
        net.fc4_5[0].bias.data.fill_(0.1)
        # fc 5
        nn.init.normal_(net.fc4_5[3].weight.data)
        net.fc4_5[3].weight.data = net.fc4_5[3].weight.data * scal.expand_as(net.fc4_5[3].weight.data)
        net.fc4_5[3].bias.data.fill_(0.1)
        # mu
        nn.init.normal_(net.mu.weight.data)
        net.mu.weight.data = net.mu.weight.data * scal.expand_as(net.mu.weight.data)
        net.mu.bias.data.fill_(0)
        # sigma
        nn.init.normal_(net.sigma.weight.data)
        net.sigma.weight.data = net.sigma.weight.data * scal.expand_as(net.sigma.weight.data)
        net.sigma.bias.data.fill_(0)
        return net

    def critic_init(self,net):
        scal = torch.Tensor([0.01])
        # fc 4
        nn.init.normal_(net.fc4_5[0].weight.data)
        net.fc4_5[0].weight.data = net.fc4_5[0].weight.data * scal.expand_as(net.fc4_5[0].weight.data)
        net.fc4_5[0].bias.data.fill_(0.1)
        # fc 5
        nn.init.normal_(net.fc4_5[3].weight.data)
        net.fc4_5[3].weight.data = net.fc4_5[3].weight.data * scal.expand_as(net.fc4_5[3].weight.data)
        net.fc4_5[3].bias.data.fill_(0.1)
        # value
        nn.init.normal_(net.value.weight.data)
        net.value.weight.data = net.value.weight.data * scal.expand_as(net.value.weight.data)
        net.value.bias.data.fill_(0)
        return net

    def Save_Model(self,path1,path2):
        torch.save(self.main_actor, path1)
        torch.save(self.critic, path2)

    def add(self,s,a,r):
        self.buffer_s.append(s)
        self.buffer_a.append(a)
        self.buffer_r.append(r)

    def perceive(self,s,a,r,s_):
        v_s_ = self.critic(s_).item()
        discounted_r = []
        for r in self.buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        bs = np.array(np.vstack(self.buffer_s))
        ba = np.array(np.vstack(self.buffer_a))  
        br = np.array(discounted_r)[:, np.newaxis]
        self.buffer_s, self.buffer_a, self.buffer_r = [], [], []
        return bs,ba,br

    def select_action(self,state):
        _,action=self.main_actor(state)
        return action

    def update(self,state,action,dc_r):
        self.target_actor=copy.deepcopy(self.main_actor)
        self.target_actor.eval()
        adv = dc_r.detach().numpy()-self.critic(state).detach().numpy()
        adv = torch.from_numpy(adv)
        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(self.A_UPDATE_STEPS):
                pi,_=self.main_actor(state)
                oldpi,_=self.target_actor(state)
                criterion=kl_pen_actor_loss()
                aloss,kl_mean=criterion(pi,oldpi,action,adv)
                self.actor_optimizer.zero_grad()
                self.main_actor.train()
                aloss.backward(retain_graph=True)
                self.actor_optimizer.step()
                self.main_actor.eval()
            if kl_mean < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl_mean > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            for _ in range(self.A_UPDATE_STEPS):
               pi,_=self.main_actor(state)
               oldpi,_=self.target_actor(state)
               criterion=clip_actor_loss()
               aloss=criterion(pi,oldpi,action,adv)

               self.main_actor.train()
               self.actor_optimizer.zero_grad()
               aloss.backward(retain_graph=True)
               self.actor_optimizer.step()
               self.main_actor.eval()   

        # update critic
        for _ in range(self.C_UPDATE_STEPS):           
            value=self.critic(state)
            print(value)
            closs =self.criterion(dc_r ,value)

            self.critic.train()
            self.critic_optimizer.zero_grad()
            closs.backward(retain_graph=True)
            self.critic_optimizer.step()
            self.critic.eval()

class clip_actor_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pi, oldpi,action,adv):
        ratio = torch.mean(torch.exp(pi.log_prob(action) - oldpi.log_prob(action)))
        clipped_ratio = torch.clamp(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])
        aloss = -torch.mean(torch.min(ratio*adv, clipped_ratio*adv))
        entropy = -torch.sum(pi.log_prob(action).exp() * torch.log(torch.clamp(pi.log_prob(action).exp(),1e-10,1.0)))
        entropy = torch.mean(entropy,axis=0)      
        aloss -= 0.001 * entropy
        return aloss

class kl_pen_actor_loss(nn.Module):
    def __init__(self):
        super().__init__()      
    def forward(self, pi, oldpi,action):
        ratio = torch.mean(torch.exp(pi.log_prob(action) - oldpi.log_prob(action)))            
        surr = ratio * adv
        kl = kl_divergence(oldpi, pi)
        kl_mean = torch.mean(kl)
        aloss = -(torch.mean(surr - METHOD['lam'] * kl))
        return aloss,kl_mean

def Train(base_network):
    if base_network == 'vggm':
        base_network = vggm()  # by default, load vggm's weights too
        # base_network = base_network.features[0:10]
        base_network = base_network.features[0:15]

    else:  # change this part if adding more base network variant
        base_network = vggm()
        # base_network = base_network.features[0:10]
        base_network = base_network.features[0:15]    
    agent=PPO(base_network,opts)
    state=torch.zeros(1,3,112,112).reshape(-1,3,112,112)
    action=torch.tensor([1.0]*13)
    dc_r=torch.tensor([[10.0]])
    agent.update(state,action,dc_r)
    print("hello")

# def str2bool(v):
#     return v.lower() in ("yes", "true", "t", "1")
#
# parser = argparse.ArgumentParser(
#     description='ADNet training')

def gym_reward(criterion,bbox,action_score):
    action=action_score[0][0:11]
    score=action_score[0][11:12]
    action=np.argmax(action.detach().cpu().numpy())
    next_bbox=do_action(bbox, opts, action, opts['imgSize'])
    IOU=cal_iou(bbox,next_bbox)
    return IOU

def adnet_train_sl(args, opts):

    # if torch.cuda.is_available():
    #     if args.cuda:
    #         torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #     if not args.cuda:
    #         print(
    #             "WARNING: It looks like you have a CUDA device, but aren't " + "using CUDA.\nRun with --cuda for optimal training speed.")
    #         torch.set_default_tensor_type('torch.FloatTensor')
    # else:
    #     torch.set_default_tensor_type('torch.FloatTensor')
    torch.set_default_tensor_type('torch.FloatTensor')

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # if args.visualize:
    #     writer = SummaryWriter(log_dir=os.path.join(args.tensorlogdir, args.save_file))
    #

    # print('generating Reinforcement Learning gym(dataset)..')

    # train_videos = get_train_videos(opts)
    # if train_videos==None:
    # train_videos = None
    # opts['num_videos'] =1
    # number_domain = opts['num_videos']
    mean = np.array(opts['means'], dtype=np.float32)
    mean = torch.from_numpy(mean).cuda()
    transform2 = ADNet_Augmentation2(opts, mean)
    # # datasets_pos, datasets_neg = initialize_pos_neg_dataset(train_videos,opts, transform=ADNet_Augmentation(opts),multidomain=args.multidomain)
    # datasets_pos_neg = initialize_pos_neg_dataset(train_videos, opts,args, transform=ADNet_Augmentation2(opts,mean),multidomain=args.multidomain)
    # # else:
    # #     opts['num_videos'] = len(train_videos['video_names'])
    # #     number_domain = opts['num_videos']
    # # datasets_pos_neg = initialize_pos_neg_dataset(train_videos, opts, args,transform=ADNet_Augmentation(opts),multidomain=args.multidomain)
    #
    # len_dataset = 0
    # for dataset_pos_neg in datasets_pos_neg:
    #     len_dataset += len(dataset_pos_neg)
    #
    # epoch_size = len_dataset // opts['minibatch_size']
    # print("1 epoch = " + str(epoch_size) + " iterations")
    #
    # data_loaders = []
    #
    # print("before  data_loaders.append(data.DataLoader(dataset_pos_neg", end=' : ')
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # for dataset_pos_neg in datasets_pos_neg:
    #     data_loaders.append(data.DataLoader(dataset_pos_neg, opts['minibatch_size'], num_workers=args.num_workers, shuffle=True, pin_memory=False))
    # print("after  data_loaders.append(data.DataLoader(dataset_pos_neg", end=' : ')
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # which_domain = np.random.permutation(number_domain)
    parser3 = argparse.ArgumentParser()
    parser3.add_argument('--eval_imgs', default=0, type=int,
                        help='the num of imgs that picked from val.txt, 0 represent all imgs')
    parser3.add_argument('--gt_skip', default=1, type=int, help='frame sampling frequency')
    parser3.add_argument('--dataset_year', default=2222, type=int,
                        help='dataset version, like ILSVRC2015, ILSVRC2017, 2222 means train.txt')
    args3 = parser3.parse_args(['--eval_imgs', '1000', '--gt_skip', '1', '--dataset_year', '2222'])
    videos_infos, _ = get_ILSVRC_eval_infos(args3)

    criterion = nn.BCELoss()
    base_network = vggm().features[0:15]
    if args.resume_actor and args.resume_critic:
        agent=PPO(base_network,opts,args.resume_actor,args.resume_critic)
    else:
        agent=PPO(base_network,opts)

    batch=32
    batch_size=32
    for epoch in range(args.start_epoch, opts['numEpoch']):
        Reward=0
        iteration=0
        for vid_info in videos_infos:
            frame_t=cv2.imread(vid_info['img_files'][0])
            opts['imgSize']=frame_t.shape[:2]
            n_frames=vid_info['nframes']
            n_objs=len(vid_info['trackid'][0])
            for obj_i in range(n_objs):
                for frame_i in range(n_frames):
                    bbox=vid_info['gt'][frame_i][obj_i]
                    image=cv2.imread(vid_info['img_files'][frame_i]).astype(np.float32)
                    image=torch.from_numpy(image).cuda()
                    # action_label=
                    # score_label=
                    if frame_i!=0 and frame_i%batch_size==0:
                        # _, nextbatch = data_sets[iteration + 1]
                        # images, _, _, _ = nextbatch
                        # next_state = images.reshape(-1, 3, 112, 112)
                        next_state, _, _, _ =transform2(image, np.array(bbox))
                        next_state=next_state.cpu()
                        bs, ba, br = agent.perceive(state, action_score, reward, next_state)
                        bs=torch.from_numpy(bs)
                        ba = torch.from_numpy(ba)
                        br = torch.from_numpy(br)
                        agent.update(bs, ba, br)
                        save_path_actor=os.path.join(args.save_folder, args.save_file_actor)+\
                                        'epoch' + repr(epoch) +"_"+ repr(iteration) +'.pth'
                        save_path_critic = os.path.join(args.save_folder, args.save_file_critic) + \
                                          'epoch' + repr(epoch) + "_" + repr(iteration) + '.pth'
                        agent.Save_Model(save_path_actor, save_path_critic)
                        iteration+=1
                        t = time.time()
                        print("Current moment:{} ;current epoch:{} ; Total Reward: {}".format(str(t),str(epoch),str(Reward)))
                        if (n_frames - frame_i) < (batch_size + 1):
                            break
                        state=next_state
                    else:
                        state, _, _, _ = transform2(image, np.array(bbox))
                        state=state.cpu()
                    action_score = agent.select_action(state)  # generate action
                    reward = gym_reward(criterion, bbox, action_score)  # get reward based on the generated action and environment
                    Reward += reward
                    agent.add(state, action_score, reward)


        #
        #
        # if args.multidomain:
        #     curr_domain = which_domain[iteration % len(which_domain)]
        # else:
        #     curr_domain = 0
        # data_sets=[]
        # for iteration, batch in enumerate(BackgroundGenerator(data_loaders[curr_domain])):
        #     data_sets.append(batch)
        # iteration=0
        # Reward=0
        # for batch in data_sets:
        #     try:
        #         images, bbox, action_label, score_label =batch
        #         images=images.reshape(-1,3,112,112)
        #         bbox=bbox.reshape(-1,4)
        #         action_label=action_label.reshape(-1,11)
        #         score_label=score_label.reshape(-1)
        #     except StopIteration:
        #         print("No  way !")
        #     state=images
        #     action_score=agent.select_action(state)#generate action
        #
        #     reward=gym_reward(criterion,bbox,action_score,action_label,score_label)#get reward based on the generated action and environment
        #     Reward+=reward
        #
        #     agent.add(state,action_score,reward)
        #     t=time.time()
        #     if (iteration+1) % batch == 0 :
        #         print('start training !')
        #         _,nextbatch=data_sets[iteration+1]
        #         images, _,_,_ =nextbatch
        #         next_state=images.reshape(-1,3,112,112)
        #         bs, ba, br=agent.perceive(state,action_score,reward,next_state)
        #         agent.update(bs, ba, br)
        #         agent.Save_Model(args.resume_actor,args.resume_critic)
        #         print("Current moment:{} ;current epoch:{}  ; current batch:{} ; Total Reward: {}".format(str(t),str(epoch),str(batch),str(Reward)))
        #     iteration+=1
        #
        #
            


if __name__ == "__main__":
    args = parser.parse_args()

    # PPO Learning part
    if args.run_ppo:
        #opts['minibatch_size'] = 128
        opts['minibatch_size'] = 256
        # args.resume_actor = os.path.join(args.save_folder, args.save_file_actor) + '.pth'
        # args.resume_critic = os.path.join(args.save_folder, args.save_file_critic) + '.pth'
        adnet_train_sl(args, opts)
    else:
        print("no else !")


