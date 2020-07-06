import _init_paths
import argparse
import copy
import glob
import os
import time
from collections import deque

import gym
import gym_vid_action
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

from options.general2 import opts
from models.ADNet import adnet
from utils.do_action import do_action
from utils.overlap_ratio import overlap_ratio
from utils.augmentations import ADNet_Augmentation2
from utils.my_util import get_ILSVRC_eval_infos

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # log_dir = os.path.expanduser(args.log_dir)
    log_dir=args.log_dir
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
    #                      args.gamma, args.log_dir, device, False)
    env = gym.make(args.env_name)

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_imgs', default=0, type=int,
                        help='the num of imgs that picked from val.txt, 0 represent all imgs')
    parser.add_argument('--gt_skip', default=1, type=int, help='frame sampling frequency')
    parser.add_argument('--dataset_year', default=2222, type=int,
                        help='dataset version, like ILSVRC2015, ILSVRC2017, 2222 means train.txt')
    args2 = parser.parse_args(['--eval_imgs', '2000', '--gt_skip', '1', '--dataset_year', '2222'])

    videos_infos, _ = get_ILSVRC_eval_infos(args2)

    mean = np.array(opts['means'], dtype=np.float32)
    mean = torch.from_numpy(mean).cuda()
    transform = ADNet_Augmentation2(opts, mean)

    # for en in envs:
    #     en.init_data(videos_infos, opts, transform, do_action,overlap_ratio)
    env.init_data(videos_infos, opts, transform, do_action, overlap_ratio)

    net, _ = adnet(opts, trained_file=args.resume, random_initialize_domain_specific=True,
                                      multidomain=False)
    # net = net.cuda()

    actor_critic = Policy(
        env.observation_space.shape,
        env.action_space,
        base=net,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    # if args.algo == 'a2c':
    #     agent = algo.A2C_ACKTR(
    #         actor_critic,
    #         args.value_loss_coef,
    #         args.entropy_coef,
    #         lr=args.lr,
    #         eps=args.eps,
    #         alpha=args.alpha,
    #         max_grad_norm=args.max_grad_norm)
    # elif args.algo == 'ppo':
    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)
    # elif args.algo == 'acktr':
    #     agent = algo.A2C_ACKTR(
    #         actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    # if args.gail:
    #     assert len(envs.observation_space.shape) == 1
    #     discr = gail.Discriminator(
    #         envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
    #         device)
    #     file_name = os.path.join(
    #         args.gail_experts_dir, "trajs_{}.pt".format(
    #             args.env_name.split('-')[0].lower()))
    #
    #     expert_dataset = gail.ExpertDataset(
    #         file_name, num_trajectories=4, subsample_frequency=20)
    #     drop_last = len(expert_dataset) > args.gail_batch_size
    #     gail_train_loader = torch.utils.data.DataLoader(
    #         dataset=expert_dataset,
    #         batch_size=args.gail_batch_size,
    #         shuffle=True,
    #         drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps,
                              opts['inputSize'], env.action_space,
                              )

    obs = env.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    # num_updates = int(
    #     args.num_env_steps) // args.num_steps // args.num_processes
    # for j in range(num_updates):
    j=-1
    while True:
        j+=1
        actor_critic.base.reset_action_dynamic()

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            # utils.update_linear_schedule(
            #     agent.optimizer, j, num_updates,
            #     agent.optimizer.lr if args.algo == "acktr" else args.lr)
            utils.update_linear_schedule(
                agent.optimizer, j, len(videos_infos),
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        # for step in range(args.num_steps):
        box_history_clip = []
        t = 0
        step=0
        while True:
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, new_state,reward, done, infos = env.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            # masks = torch.FloatTensor(
            #     [[0.0] if done_ else [1.0] for done_ in done])
            # bad_masks = torch.FloatTensor(
            #     [[0.0] if 'bad_transition' in info.keys() else [1.0]
            #      for info in infos])
            masks = torch.FloatTensor(
                [1.0])
            bad_masks = torch.FloatTensor(
                [1.0])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

            if ((action != opts['stop_action']) and any(
                    (np.array(new_state).round() == x).all() for x in np.array(box_history_clip).round())):
                action = opts['stop_action']
                reward, done, finish_epoch = env.go_to_next_frame()
                infos['finish_epoch'] = finish_epoch

            if t > opts['num_action_step_max']:
                action = opts['stop_action']
                reward, done, finish_epoch = env.go_to_next_frame()
                infos['finish_epoch'] = finish_epoch

            box_history_clip.append(list(new_state))

            t += 1

            if action == opts['stop_action']:#finish one frame
                t = 0
                box_history_clip = []
                rollouts.obs[rollouts.get_step()].copy_(env.get_current_patch)

            if done:  # if finish the clip
                rollouts.obs[rollouts.get_step()].copy_(obs)
                break

        with torch.no_grad():
            # next_value = actor_critic.get_value(
            #     rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
            #     rollouts.masks[-1]).detach()
            next_value = actor_critic.get_value(
                rollouts.obs[rollouts.get_step()], rollouts.recurrent_hidden_states[rollouts.get_step()],
                rollouts.masks[rollouts.get_step()]).detach()

        # if args.gail:
        #     if j >= 10:
        #         envs.venv.eval()
        #
        #     gail_epoch = args.gail_epoch
        #     if j < 10:
        #         gail_epoch = 100  # Warm up
        #     for _ in range(gail_epoch):
        #         discr.update(gail_train_loader, rollouts,
        #                      utils.get_vec_normalize(envs)._obfilt)
        #
        #     for step in range(args.num_steps):
        #         rollouts.rewards[step] = discr.predict_reward(
        #             rollouts.obs[step], rollouts.actions[step], args.gamma,
        #             rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.obs[rollouts.get_step()].copy_(env.get_current_patch)

        rollouts.after_update()

        if infos['finish_epoch']:
            break

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
