import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from torch.distributions import kl_divergence
import copy

import torch.optim as optim
from models.vggm import vggm
from adnet.options.general2 import opts
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

        if self.use_gpu:
            self.action_dynamic = self.action_dynamic.cuda()
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

        if self.use_gpu:
            self.action_dynamic = self.action_dynamic.cuda()
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
    def __init__(self, base_network, opts, num_classes=11, phase='pi', num_history=10, use_gpu=True):
        self.main_actor=Actor(base_network, opts)
        self.target_actor=Actor(base_network, opts)
        self.critic=Critic(base_network, opts)
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
    def add(self,s,a,r):
        self.buffer_s.append(s)
        self.buffer_a.append(a)
        self.buffer_r.append(r) 
    def perceive(self,s,a,r,s_):
        v_s_ = self.critic(s_)
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
        adv = dc_r-self.critic(state)
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

def adnet(opts, base_network='vggm', trained_file=None, random_initialize_domain_specific=False, multidomain=True):

    assert base_network in ['vggm'], "Base network variant is unavailable"

    num_classes = opts['num_actions']
    num_history = opts['num_action_history']

    assert num_classes in [11], "num classes is not exist"

    settings = pretrained_settings['adnet']

    if base_network == 'vggm':
        base_network = vggm()  # by default, load vggm's weights too
        # base_network = base_network.features[0:10]
        base_network = base_network.features[0:15]

    else:  # change this part if adding more base network variant
        base_network = vggm()
        # base_network = base_network.features[0:10]
        base_network = base_network.features[0:15]

    if trained_file:
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        print('Resuming training, loading {}...'.format(trained_file))

        adnet_model = ADNet(base_network=base_network, opts=opts, num_classes=num_classes, num_history=num_history)

        adnet_model.load_weights(trained_file)

        adnet_model.input_space = settings['input_space']
        adnet_model.input_size = settings['input_size']
        adnet_model.input_range = settings['input_range']
        adnet_model.mean = settings['mean']
        adnet_model.std = settings['std']
    else:
        adnet_model = ADNet(base_network=base_network, opts=opts, num_classes=num_classes)


    return adnet_model


torch.set_default_tensor_type('torch.cuda.FloatTensor')
Train('vggm')