# matlab code:
# https://github.com/hellbell/ADNet/blob/master/train/adnet_train_SL.m
# reference: https://github.com/amdegroot/ssd.pytorch/blob/master/train.py

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from models.ADNet import adnet
from utils.get_train_videos import get_train_videos
from datasets.sl_dataset import initialize_pos_neg_dataset
from utils.augmentations import ADNet_Augmentation,ADNet_Augmentation2

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from random import shuffle

import os
import time
import numpy as np

from tensorboardX import SummaryWriter


def adnet_train_sl(args, opts):

    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print(
                "WARNING: It looks like you have a CUDA device, but aren't " + "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if args.visualize:
        writer = SummaryWriter(log_dir=os.path.join('tensorboardx_log', args.save_file))


    print('generating Supervised Learning dataset..')

    train_videos = get_train_videos(opts)
    if train_videos==None:
        opts['num_videos'] =1
        number_domain = opts['num_videos']
        mean = np.array(opts['means'], dtype=np.float32)
        mean = torch.from_numpy(mean).cuda()
        # datasets_pos, datasets_neg = initialize_pos_neg_dataset(train_videos,opts, transform=ADNet_Augmentation(opts),multidomain=args.multidomain)
        datasets_pos_neg = initialize_pos_neg_dataset(train_videos, opts, transform=ADNet_Augmentation2(opts,mean),multidomain=args.multidomain)
    else:
        opts['num_videos'] = len(train_videos['video_names'])
        number_domain = opts['num_videos']
        # datasets_pos, datasets_neg = initialize_pos_neg_dataset(train_videos, opts, transform=ADNet_Augmentation(opts),multidomain=args.multidomain)
        datasets_pos_neg = initialize_pos_neg_dataset(train_videos, opts, transform=ADNet_Augmentation(opts),multidomain=args.multidomain)

        # dataset = SLDataset(train_videos, opts, transform=

    net, domain_specific_nets = adnet(opts=opts, trained_file=args.resume, multidomain=args.multidomain)

    if args.cuda:
        net = nn.DataParallel(net)
        cudnn.benchmark = True

        net = net.cuda()

    if args.cuda:
        optimizer = optim.SGD([
            {'params': net.module.base_network.parameters(), 'lr': 1e-4},
            {'params': net.module.fc4_5.parameters()},
            {'params': net.module.fc6.parameters()},
            {'params': net.module.fc7.parameters()}],  # as action dynamic is zero, it doesn't matter
            lr=1e-3, momentum=opts['train']['momentum'], weight_decay=opts['train']['weightDecay'])
    else:
        optimizer = optim.SGD([
            {'params': net.base_network.parameters(), 'lr': 1e-4},
            {'params': net.fc4_5.parameters()},
            {'params': net.fc6.parameters()},
            {'params': net.fc7.parameters()}],
            lr=1e-3, momentum=opts['train']['momentum'], weight_decay=opts['train']['weightDecay'])

    if args.resume:
        # net.load_weights(args.resume)
        checkpoint = torch.load(args.resume)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    net.train()


    if not args.resume:
        print('Initializing weights...')

        if args.cuda:
            scal = torch.Tensor([0.01])
            # fc 4
            nn.init.normal_(net.module.fc4_5[0].weight.data)
            net.module.fc4_5[0].weight.data = net.module.fc4_5[0].weight.data * scal.expand_as(net.module.fc4_5[0].weight.data)
            net.module.fc4_5[0].bias.data.fill_(0.1)
            # fc 5
            nn.init.normal_(net.module.fc4_5[3].weight.data)
            net.module.fc4_5[3].weight.data = net.module.fc4_5[3].weight.data * scal.expand_as(net.module.fc4_5[3].weight.data)
            net.module.fc4_5[3].bias.data.fill_(0.1)

            # fc 6
            nn.init.normal_(net.module.fc6.weight.data)
            net.module.fc6.weight.data = net.module.fc6.weight.data * scal.expand_as(net.module.fc6.weight.data)
            net.module.fc6.bias.data.fill_(0)
            # fc 7
            nn.init.normal_(net.module.fc7.weight.data)
            net.module.fc7.weight.data = net.module.fc7.weight.data * scal.expand_as(net.module.fc7.weight.data)
            net.module.fc7.bias.data.fill_(0)
        else:
            scal = torch.Tensor([0.01])
            # fc 4
            nn.init.normal_(net.fc4_5[0].weight.data)
            net.fc4_5[0].weight.data = net.fc4_5[0].weight.data * scal.expand_as(net.fc4_5[0].weight.data )
            net.fc4_5[0].bias.data.fill_(0.1)
            # fc 5
            nn.init.normal_(net.fc4_5[3].weight.data)
            net.fc4_5[3].weight.data = net.fc4_5[3].weight.data * scal.expand_as(net.fc4_5[3].weight.data)
            net.fc4_5[3].bias.data.fill_(0.1)
            # fc 6
            nn.init.normal_(net.fc6.weight.data)
            net.fc6.weight.data = net.fc6.weight.data * scal.expand_as(net.fc6.weight.data)
            net.fc6.bias.data.fill_(0)
            # fc 7
            nn.init.normal_(net.fc7.weight.data)
            net.fc7.weight.data = net.fc7.weight.data * scal.expand_as(net.fc7.weight.data)
            net.fc7.bias.data.fill_(0)

    action_criterion = nn.CrossEntropyLoss()
    score_criterion = nn.CrossEntropyLoss()

    # batch_iterators_pos = []
    # batch_iterators_neg = []

    # calculating number of data
    # len_dataset_pos = 0
    # len_dataset_neg = 0
    len_dataset = 0
    for dataset_pos_neg in datasets_pos_neg:
        len_dataset += len(dataset_pos_neg)
    # for dataset_neg in datasets_neg:
    #     len_dataset_neg += len(dataset_neg)

    # epoch_size_pos = len_dataset_pos // opts['minibatch_size']
    # epoch_size_neg = len_dataset_neg // opts['minibatch_size']
    # epoch_size = epoch_size_pos + epoch_size_neg  # 1 epoch, how many iterations
    epoch_size = len_dataset // opts['minibatch_size']
    print("1 epoch = " + str(epoch_size) + " iterations")

    # max_iter = opts['numEpoch'] * epoch_size
    # print("maximum iteration = " + str(max_iter))

    data_loaders = []
    # data_loaders_neg = []

    print("before  data_loaders.append(data.DataLoader(dataset_pos_neg", end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for dataset_pos_neg in datasets_pos_neg:
        data_loaders.append(data.DataLoader(dataset_pos_neg, opts['minibatch_size'], num_workers=args.num_workers, shuffle=True, pin_memory=False))
    # for dataset_neg in datasets_neg:
    #     data_loaders_neg.append(data.DataLoader(dataset_neg, opts['minibatch_size'], num_workers=args.num_workers, shuffle=True, pin_memory=True))
    print("after  data_loaders.append(data.DataLoader(dataset_pos_neg", end=' : ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # epoch = args.start_epoch
    # if epoch != 0 and args.start_iter == 0:
    #     start_iter = epoch * epoch_size
    # else:
    #     start_iter = args.start_iter

    # which_dataset = list(np.full(epoch_size_pos, fill_value=1))
    # which_dataset.extend(np.zeros(epoch_size_neg, dtype=int))
    # shuffle(which_dataset)

    which_domain = np.random.permutation(number_domain)

    action_loss = 0
    score_loss = 0

    # training loop
    # for iteration in range(start_iter, max_iter):
    # iteration=0
    # curr_domain = 0
    for epoch in range(args.start_epoch, opts['numEpoch']):
        ave_loss=0
        if epoch == args.start_epoch:
            t1 = time.time()
            t4 = time.time()
        # if new epoch (not including the very first iteration)
        if (epoch != args.start_epoch) :
            # epoch += 1
            # shuffle(which_dataset)
            # np.random.shuffle(which_domain)

            print('Saving state, epoch:', epoch-1)
            domain_specific_nets_state_dict = []
            for domain_specific_net in domain_specific_nets:
                domain_specific_nets_state_dict.append(domain_specific_net.state_dict())

            torch.save({
                'epoch': epoch-1,
                'adnet_state_dict': net.state_dict(),
                'adnet_domain_specific_state_dict': domain_specific_nets,
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.save_folder, args.save_file) +
                       'epoch' + repr(epoch-1) + '_final.pth')

            if args.visualize:
                writer.add_scalars('data/epoch_loss', {'action_loss': action_loss / epoch_size,
                                                       'score_loss': score_loss / epoch_size,
                                                       'total': (action_loss + score_loss) / epoch_size}, global_step=epoch-1)

            # reset epoch loss counters
            action_loss = 0
            score_loss = 0

        # if new epoch (including the first iteration), initialize the batch iterator
        # or just resuming where batch_iterator_pos and neg haven't been initialized
        # if iteration % epoch_size == 0 or len(batch_iterators_pos) == 0 or len(batch_iterators_neg) == 0:
            # create batch iterator
            # batch_iterators_pos = []
            # batch_iterators_neg = []
            # print("before batch_iterators_pos.append(iter(data_loader_pos", end=' : ')
            # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            # for data_loader_pos in data_loaders_pos:
            #     batch_iterators_pos.append(iter(data_loader_pos))
            # for data_loader_neg in data_loaders_neg:
            #     batch_iterators_neg.append(iter(data_loader_neg))
            # print("after batch_iterators_neg.append(iter(data_loader_neg", end=' : ')
            # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # if not batch_iterators_pos[curr_domain]:
        #     # create batch iterator
        #     batch_iterators_pos[curr_domain] = iter(data_loaders_pos[curr_domain])
        #
        # if not batch_iterators_neg[curr_domain]:
        #     # create batch iterator
        #     batch_iterators_neg[curr_domain] = iter(data_loaders_neg[curr_domain])

        # load train data
        # if which_dataset[iteration % len(which_dataset)]:  # if positive
        #     try:
        #         # print("before next(batch_iterators_pos", end=' : ')
        #         # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        #         images, bbox, action_label, score_label, vid_idx = next(batch_iterators_pos[curr_domain])
        #         # print("after next(batch_iterators_pos", end=' : ')
        #         # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        #     except StopIteration:
        #         batch_iterators_pos[curr_domain] = iter(data_loaders_pos[curr_domain])
        #         images, bbox, action_label, score_label, vid_idx = next(batch_iterators_pos[curr_domain])
        # else:
        #     try:
        #         # print("before next(batch_iterators_neg", end=' : ')
        #         # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        #         images, bbox, action_label, score_label, vid_idx = next(batch_iterators_neg[curr_domain])
        #         # print("after next(batch_iterators_neg", end=' : ')
        #         # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        #     except StopIteration:
        #         batch_iterators_neg[curr_domain] = iter(data_loaders_neg[curr_domain])
        #         images, bbox, action_label, score_label, vid_idx = next(batch_iterators_neg[curr_domain])

        batch_iterators = []
        for data_loader in data_loaders:
            batch_iterators.append(iter(data_loader))
        for iteration in range(epoch_size):
            if args.multidomain:
                curr_domain = which_domain[iteration % len(which_domain)]
            else:
                curr_domain = 0
            try:
                images, bbox, action_label, score_label, vid_idx = next(batch_iterators[curr_domain])
                images=images.reshape(-1,3,112,112)
                bbox=bbox.reshape(-1,4)
                action_label=action_label.reshape(-1,11)
                score_label=score_label.reshape(-1)
                vid_idx=vid_idx.reshape(-1)
            except StopIteration:
                batch_iterators[curr_domain] = iter(data_loader[curr_domain])
                images, bbox, action_label, score_label, vid_idx = next(batch_iterators[curr_domain])
                images = images.reshape(-1, 3, 112, 112)
                bbox = bbox.reshape(-1, 4)
                action_label = action_label.reshape(-1, 11)
                score_label = score_label.reshape(-1)
                vid_idx = vid_idx.reshape(-1)
        # for images, bbox, action_label, score_label, vid_idx in data_loaders[curr_domain]:
            # TODO: check if this requires grad is really false like in Variable
            pos_idx=torch.where(score_label>0.3)
            pos_idx=pos_idx[0].tolist()
            # pos_idx=ids.nonzero()
            # neg_idx=torch.where(score_label<=0.3)
            # neg_idx=neg_idx[0].tolist()
            # if args.cuda:
            #     images = torch.Tensor(images.cuda())
            #     bbox = torch.Tensor(bbox.cuda())
            #     action_label = torch.Tensor(action_label.cuda())
            #     score_label = torch.Tensor(score_label.float().cuda())
            #
            # else:
            #     images = torch.Tensor(images)
            #     bbox = torch.Tensor(bbox)
            #     action_label = torch.Tensor(action_label)
            #     score_label = torch.Tensor(score_label)

            t0 = time.time()

            # load ADNetDomainSpecific with video index
            if args.cuda:
                net.module.load_domain_specific(domain_specific_nets[curr_domain])
            else:
                net.load_domain_specific(domain_specific_nets[curr_domain])

            # forward
            # print("before forward net", end=' : ')
            # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            action_out, score_out = net(images)
            # print("after forward net", end=' : ')
            # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            # backprop
            optimizer.zero_grad()
            # if which_dataset[iteration % len(which_dataset)]:  # if positive
            #     action_l = action_criterion(action_out, torch.max(action_label, 1)[1])
            # else:
            #     action_l = torch.Tensor([0])
            action_l = action_criterion(action_out[pos_idx], torch.max(action_label[pos_idx], 1)[1])
            score_l = score_criterion(score_out, score_label.long())
            loss = action_l + score_l
            loss.backward()
            optimizer.step()

            action_loss += action_l.item()
            score_loss += score_l.item()
            ave_loss+=loss.data.item()

            # save the ADNetDomainSpecific back to their module
            if args.cuda:
                domain_specific_nets[curr_domain].load_weights_from_adnet(net.module)
            else:
                domain_specific_nets[curr_domain].load_weights_from_adnet(net)

            #t1 = time.time()

            if iteration % 2000 == 0 and iteration!=0:
                #print('Timer: %.4f sec.' % (t1 - t0))
                #print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data.item()), end=' ')
                #if iteration==start_iter:
                    # t4=time.time()
                t5=time.time()
                #t3='Timer: %.4f sec.' % (t5 - t4)
                t3=t5-t4
                t3_m = t3 // 60
                t3_s = t3 % 60

                all_time = t5 - t1
                all_d = all_time//86400
                all_h = all_time %86400 // 3600
                all_m = all_time % 3600 // 60
                all_s = all_time % 60
                t4=time.time()
                ave_loss=ave_loss/2000
                print('epoch '+repr(epoch)+' | iter ' + repr(iteration) + ' | L-now: %.4f | L-ave: %.4f | T-iter: %d m %d s | T-all: %d d %d h %d m %d s | T-now: ' % (loss.data.item(),ave_loss,t3_m,t3_s,all_d,all_h,all_m,all_s),end='')
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                ave_loss=0

                if args.visualize and args.send_images_to_visualization:
                    random_batch_index = np.random.randint(images.size(0))
                    writer.add_image('image', images.data[random_batch_index].cpu().numpy(), random_batch_index)

            if args.visualize:
                writer.add_scalars('data/iter_loss', {'action_loss': action_l.item(),
                                                      'score_loss': score_l.item(),
                                                      'total': (action_l.item() + score_l.item())}, global_step=iteration)
                # hacky fencepost solution for 0th epoch plot
                if iteration == 0:
                    writer.add_scalars('data/epoch_loss', {'action_loss': action_loss,
                                                           'score_loss': score_loss,
                                                           'total': (action_loss + score_loss)}, global_step=epoch)

            if iteration % 5000 == 0 and iteration!=0:
                print('Saving state, iter:', iteration)

                domain_specific_nets_state_dict = []
                for domain_specific_net in domain_specific_nets:
                    domain_specific_nets_state_dict.append(domain_specific_net.state_dict())

                torch.save({
                    'epoch': epoch,
                    'adnet_state_dict': net.state_dict(),
                    'adnet_domain_specific_state_dict': domain_specific_nets,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(args.save_folder, args.save_file) +
                           'epoch' + repr(epoch) +"_"+ repr(iteration) +'.pth')
            # iteration = iteration + 1
            # if args.multidomain:
            #     curr_domain = which_domain[iteration % len(which_domain)]
            # else:
            #     curr_domain = 0
    # final save
    torch.save({
        'epoch': opts['numEpoch']-1,
        'adnet_state_dict': net.state_dict(),
        'adnet_domain_specific_state_dict': domain_specific_nets,
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.save_folder, args.save_file) + '.pth')

    return net, domain_specific_nets, train_videos



