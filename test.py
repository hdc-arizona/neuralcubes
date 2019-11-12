import sys
import numpy as np
import math
import argparse
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as LRS

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# import matplotlib.pyplot as plt

from denet_0_4 import *
# from tensor_blowup import TensorBlowup_YellowCab
from tensor_blowup import TensorBlowup
from utils import *

def test(modelPath, dataFilePath):
    net = torch.load(modelPath)
    # net = pickle.load(open(modelPath, 'rb'))
    if args.cuda:
        net.cuda()
    net.cpu()
    net.eval()

    data_schema = net.schema()["data_schema"]
    net_schema = net.schema()["net_schema"]

    # get dimensions
    dims = []
    for b in net_schema["branches"]:
        inputSize = -1
        if data_schema[b["key"]]["type"] == "spatial":
            res = data_schema[b["key"]]["resolution"]
            inputSize = res["x"]+res["y"]
        else:
            inputSize = data_schema[b["key"]]["dimension"]
        dims.append(inputSize)

    # plot heatmaps of weight with pyplot
    # net.apply(vis)

    test_dataset = DENetRangesDataset(dataFilePath, schema=net.schema())
    testloader = DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=4)

    criterion1 = nn.BCELoss()
    criterion2 = nn.L1Loss()
    if args.cuda:
        criterion1.cuda()
        criterion2.cuda()

    results = []

    hm_dist_list = [ [] for i in range(len(dims))]
    running_loss1 = 0.0
    running_loss2 = 0.0

    def norm_hamming(p, t):
        p[p < 0.5] = 0
        p[p > 0.5] = 1
        return np.sum(p != t) / t.shape[0]

    # tensor blow-up
    # tb = TensorBlowup_YellowCab()
    tb = TensorBlowup(schema=net.schema())

    for data in testloader:
        inputs, labels = data['ranges'], data['counts']
        # blow-up
        inputs = tb.blowup(inputs)
        labels = labels.view((-1, 1))

        b_size = inputs.size()[0]

        if args.cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs).float(), Variable(labels).float()

        out_ae, out_reg = net(inputs)

        for i in range(b_size):
            predict = out_reg[i].item()
            truth = labels[i].item()
            if args.noclip:
                results.append((predict, truth))
            else:
                results.append((max(0,predict), truth))

            # print((predict, truth))

            # startLoc = 0
            # for k in range(len(dims)):
                # hm_dist_list[k].append(
                    # norm_hamming(np.array(out_ae[k][i].data.cpu()), np.array(inputs[i].data.cpu()[startLoc:startLoc+dims[k]])))
                # startLoc += dims[k]

    # print(60*'-')
    stats(results)


def stats(results):
    results = np.array(results).astype(np.float64)

    np.set_printoptions(precision=3, suppress=True)

    prediction = results[:,0]
    truth = results[:,1]
    # print('max prediction:', np.max(prediction))
    # print('max truth:', np.max(truth))
    # print('min prediction:', np.min(prediction))
    # print('min truth:', np.min(truth))
    # print('mean prediction:', np.mean(prediction))
    # print('mean truth:', np.mean(truth))
    # print('variance truth - prediction:', np.var(truth-prediction))
    abs_err = np.absolute(truth - prediction)
    # rel_err = abs_err[truth!=0] / truth[truth!=0]
    # abs_err_tm = abs_err/ np.mean(truth)
    # abs_err_pm = abs_err/ np.mean(prediction)
    # print('abs err / mean(truth):', np.mean(abs_err_tm))
    # print('abs err / mean(pred):', np.mean(abs_err_pm))
    # print('Mean Absolute Error(MAE):', np.mean(abs_err))
    # print('mean relative error:', np.mean(rel_err))
    sum_abs_err = np.sum(abs_err)
    sum_variance = np.sum(np.absolute(truth - np.mean(truth)))
    rae = sum_abs_err / sum_variance
    print('Relative Absolute Error(RAE):', rae)

    # plot histogram of individual RAE
    # n = truth.size
    # values = n*abs_err.astype(float) / sum_variance.astype(float)
    # bins = np.linspace(0, 1, 30)
    # plt.hist(values, bins=bins)
    # plt.yscale('log', nonposy='clip')
    # plt.ylim(10, 1000000)
    # plt.axis('off')
    # plt.xlabel('error', fontsize=40)
    # plt.savefig('test_hist.png', bbox_inches='tight', pad_inches = 0)

    # plt.scatter(truth, values, alpha=0.2)
    # plt.savefig('test_scatter.png')

    # r2 = 1 - np.sum(np.square(abs_err)) / np.sum(np.square(truth-np.mean(truth)))
    # print('R2 Score: ', r2)


    #hist_p,edge_p = np.histogram(prediction[truth!=0],bins=10)
    #hist_t,edge_t = np.histogram(truth[truth!=0],bins=10)
    #x = np.arange(truth.shape[0])
    #from matplotlib import gridspec
    #fig = plt.figure()
    #gs = gridspec.GridSpec(2,4)
    #phist = fig.add_subplot(gs[0,0]);
    #thist = fig.add_subplot(gs[1,0],sharex=phist);
    #adiff = fig.add_subplot(gs[:,1:]);
    #phist.bar(edge_p[:-1],hist_p,np.diff(edge_p),color="pink",ec='k',align='edge');
    #phist.set_title("histogram of prediction")
    #thist.bar(edge_t[:-1],hist_t,np.diff(edge_t),color="orange",ec='k',align='edge');
    #thist.set_title("histogram of truth")
    #adiff.plot(x,truth,'pink', x, prediction,'m')
    #adiff.set_title("truth - pred")
    #adiff.set_xlabel("x = entries")
    #plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing Neural Networks for Data Visualization.')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--noclip', default=False, action='store_true', help='clip negtive predictions or not')
    parser.add_argument('-f', nargs='?', help='testing file path')
    parser.add_argument('-m', nargs='?', help='saved model filename')

    args = parser.parse_args()
    test(args.m, args.f)

