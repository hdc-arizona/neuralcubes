import sys
import numpy as np
import math
import argparse
import json
import os
import subprocess
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as LRS

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from denet_0_4 import *
# from tensor_blowup import TensorBlowup_YellowCab
from tensor_blowup import TensorBlowup
from utils import *
import time


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        if opt.deep_flight_init:
            m.weight.data.normal_(0.05, 0.21)
            # nn.init.xavier_normal(m.weight)
        elif opt.flight_init:
            m.weight.data.normal_(0.13, 0.45)
            # nn.init.xavier_normal(m.weight)
        else:
            m.weight.data.normal_(opt.mu, opt.sigma)
        m.bias.data.fill_(0)
    # elif classname.find('BatchNorm') != -1:
        # m.weight.data.normal_(1.0, 0.02)
        # m.bias.data.fill_(0)


def train(cfg, opt):
    # Initialize network with schemas
    net = DENet(cfg["data_schema"], cfg["net_schema"])

    meta_data = cfg["net_schema"]["meta-data"]
    if opt.cuda:
        train
        net.cuda()
        net.logMeta("Use GPU", "Yes")

    net.apply(weights_init)

    # Define loss functions
    criterion_BCE = nn.BCELoss()
    # criterion_L1 = nn.L1Loss(size_average=True)
    # criterion_MSE = nn.MSELoss(size_average=True)

    def criterion_L1(t1, t2, weights=1):
        l1 = torch.abs(out_reg-labels)
        if opt.weighted:
            l1 *= weights
        return torch.mean(l1)

    def criterion_MSE(t1, t2, weights=1):
        mse = torch.pow(out_reg-labels, 2)
        if opt.weighted:
            mse *= weights
        return torch.mean(mse)

    AE_loss_weight = meta_data["ae_loss_weight"]
    REG_l1loss_weight = meta_data["reg_L1Loss_weight"]
    REG_mseloss_weight = meta_data["reg_MSELoss_weight"]
    regularization_weight = meta_data["regularization_weight"] if "regularization_weight" in meta_data else 0

    if opt.cuda:
        criterion_BCE.cuda()
        # criterion_L1.cuda()
        # criterion_MSE.cuda()

    optimizer_sgd = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum)
    # optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    net.logMeta("Optimizer", "optim.SGD(net.parameters(), lr=%f, momentum=%f)" % (opt.lr, opt.momentum))

    optimizer_adam = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    net.logMeta("Optimizer", "optim.Adam(net.parameters(), lr=%f, betas=(0.5, 0.999))" % (opt.lr))

    if opt.plateau:
        # scheduler = LRS.StepLR(optimizer, 5, gamma=0.5)
        scheduler = LRS.ReduceLROnPlateau(optimizer_sgd, mode='min', factor=0.2, patience=8, threshold=0.001, verbose=True)
    net.logMeta("Learning Rate Scheduler", "LRS.ReduceLROnPlateau(optimizer, 'min', verbose=True)")

    ###########################################################################
    # Load training data
    filePath = urlparse(meta_data['training_set'])
    remoteLoc = False
    if filePath.scheme != '':
        filePath = os.path.basename(filePath.path)  # this is a url, we find it locally first
        remoteLoc = True
    else:
        filePath = filePath.path

    if not os.path.exists(filePath):
        if remoteLoc is True:
            print("Downloading training set from {}".format(meta_data['training_set']))
            subprocess.call(["curl", "-O", meta_data['training_set']])
        else:
            print("Can not find training set {}".format(meta_data['training_set']))
            exit(0)
    print('Using training set at {}'.format(filePath))

    nameSplit = os.path.splitext(filePath)
    if nameSplit[1] == '.txt':
        filePath = filePath
    elif nameSplit[1] == '.gz':
        if not os.path.exists(nameSplit[0]):
            subprocess.call(["gunzip", '-k', filePath])
        filePath = nameSplit[0]
    else:
        optopt
        print("Training set file type not supported.")
        exit(0)

    print("Loading data...")
    # bk_dataset = DENetRangesDataset(filePath, transform=transforms.Compose([ToTensor('float')]))
    bk_dataset = DENetRangesDataset(filePath, schema=cfg)
    dataloader = DataLoader(bk_dataset, batch_size=meta_data["mini_batch_size"], shuffle=True, num_workers=4)

    ###########################################################################
    # Start training
    net.train()

    num_params = 0
    for layer in net.parameters():
        num_params += layer.numel()

        # get dimensions
    data_schema = cfg["data_schema"]
    net_schema = cfg["net_schema"]
    dims = []
    for b in net_schema["branches"]:
        inputSize = -1
        if data_schema[b["key"]]["type"] == "spatial":
            res = data_schema[b["key"]]["resolution"]
            inputSize = res["x"] + res["y"]
        else:
            inputSize = data_schema[b["key"]]["dimension"]
        dims.append(inputSize)
    logfile = open(opt.outf + "training.log", 'w')
    logfile.write('#N_PARAM %d\n' % num_params)

    # tensor blow-up
    # tb = TensorBlowup_YellowCab()
    tb = TensorBlowup(schema=cfg)

    # whether switched from Adam to SGD
    switched = False
    for epoch in range(meta_data["training_epoch"]):
        if epoch >= opt.switch:
            switched = True

        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0
        running_regularizer = 0.0
        running_total = 0.0
        tick = time.time()
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data['ranges'], data['counts']

            # blow-up
            inputs = tb.blowup(inputs)
            labels = labels.view((-1, 1))
            weights = tb.createWeights(meta_data["mini_batch_size"])

            # count regularization data
            if opt.count_regularizer:
                mh_split_1,mh_split_2,sampled_inds = tb.random_count_split(inputs,100)
                sum_labels = labels[sampled_inds,:]

            if opt.cuda:
                inputs, labels, weights = inputs.cuda(), labels.cuda(), weights.cuda()
                if opt.count_regularizer:
                    mh_split_1,mh_split_2,sum_labels = mh_split_1.cuda(),mh_split_2.cuda(),sum_labels.cuda()
            inputs, labels = inputs.float(), labels.float()
            if opt.count_regularizer:
                mh_split_1,mh_split_2,sum_labels = mh_split_1.float(),mh_split_2.float(),sum_labels.float()

            if switched:
                optimizer_sgd.zero_grad()
            else:
                optimizer_adam.zero_grad()

            # main objective
            out_ae, out_reg = net(inputs)

            # regularizer
            if opt.count_regularizer:
                _,out_split_1 = net(mh_split_1)
                _,out_split_2 = net(mh_split_2)

            # calculate AE losses
            startLoc = 0
            loss_ae = []
            for i in range(len(out_ae)):
                loss_ae.append(criterion_BCE(out_ae[i], inputs[:, startLoc: startLoc + dims[i]]))
                startLoc += dims[i]
            lossAE = sum(loss_ae)

            # loss of regressor
            lossRegressor_l1 = criterion_L1(out_reg, labels, weights)
            lossRegressor_mse = criterion_MSE(out_reg, labels, weights)

            # regularization term
            if opt.count_regularizer:
                count_regularizer = criterion_MSE((out_split_1+out_split_2),sum_labels)

            totalloss = AE_loss_weight * lossAE + REG_l1loss_weight * lossRegressor_l1 + REG_mseloss_weight * lossRegressor_mse
            if opt.count_regularizer:
                totalloss += regularization_weight*count_regularizer
            totalloss.backward()

            if switched:
                optimizer_sgd.step()
            else:
                optimizer_adam.step()

            running_loss1 += lossAE.item()/len(dataloader)
            running_loss2 += lossRegressor_l1.item()/len(dataloader)
            running_loss3 += lossRegressor_mse.item()/len(dataloader)
            if opt.count_regularizer:
                running_regularizer += count_regularizer.item()/len(dataloader)

            running_total += totalloss.item()/len(dataloader)
        tock = time.time()
        print('time to train an epoch:',(tock-tick))

        print('[{}] AE loss: {:.3f}({:.3f}), Regressor L1 loss: {:.3f}({:.3f}), L2 loss: {:.3f}({:.3f}), Count Regularizer: {:.3f}({:.3f})'.format(
            epoch,
            running_loss1 , running_loss1 * AE_loss_weight,
            running_loss2 , running_loss2 * REG_l1loss_weight,
            running_loss3 , running_loss3 * REG_mseloss_weight,
            running_regularizer , running_regularizer * regularization_weight)
        )
        if opt.plateau and switched:
            # scheduler.step()
            scheduler.step(running_total)

        logfile.write("%d, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n" % (
            epoch,
            running_loss1 , running_loss1  * AE_loss_weight,
            running_loss2 , running_loss2  * REG_l1loss_weight,
            running_loss3 , running_loss3  * REG_mseloss_weight,
            running_regularizer , running_regularizer  * regularization_weight,
        ))
        net.logMeta("current_epoch", epoch)
        torch.save(net, opt.outf + "checkpoints/epoch%d.pth" % epoch)

    # save as cpu model
    net.cpu()
    torch.save(net, opt.outf + "checkpoints/_net.pth")
    logfile.close()


def direxists(path):
    return os.path.exists(path) and os.path.isdir(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Neural Networks.')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('-c', nargs='?', default='./config.json', help='Configure file')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for lr scheduler')
    parser.add_argument('--flight_init', action='store_true',
                        help='Use weight initializations designed for flights 2008 dataset')
    parser.add_argument('--deep_flight_init', action='store_true',
                        help='Use weight initializations designed for deep architecture of flights 2008 dataset')
    parser.add_argument('--outf', required=True, help='output folder')
    parser.add_argument('--name', required=True, help='jobname')
    parser.add_argument('--mu', default = 0, type=float, help='linear weight init mu for normal_, default=0.0')
    parser.add_argument('--sigma', default=0.03, type=float, help='linear weight init sigma for normal_, default=0.03')

    parser.add_argument('--switch', default=0, type=int, help='switch from Adam to SGD after epochs, 0(default) means always use SGD')

    parser.add_argument('--weighted', dest='weighted', action='store_true', help='use weights when calculating loss')
    parser.set_defaults(weighted=False)

    parser.add_argument('--plateau', dest='plateau', action='store_true', help='enables plateau scheduler')
    parser.add_argument('--no-plateau', dest='plateau', action='store_false', help='disables plateau scheduler')
    parser.set_defaults(plateau=True)

    parser.add_argument('--count-regularizer', dest='count_regularizer', action='store_true', help='enables count regularizer')
    parser.add_argument('--no-count-regularizer', dest='count_regularizer', action='store_false', help='disables count regularizer')
    parser.set_defaults(count_regularizer=False)

    opt = parser.parse_args()
    # initialize opt
    if opt.outf[-1] != '/':
        opt.outf += '/'
    opt.outf += opt.name + '/'
    checkdir = opt.outf + 'checkpoints/'

    if not direxists(opt.outf):
        os.makedirs(opt.outf)
    if not direxists(checkdir):
        os.makedirs(checkdir)

    cfg = json.load(open(opt.c))

    tstart = time.time()
    train(cfg, opt)
    telapsed = time.time() - tstart

    with open(opt.outf + 'training.log', 'a') as logfile:
        logfile.write("#TIME %f" % telapsed)
