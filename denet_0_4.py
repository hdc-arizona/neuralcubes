import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as LRS

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class DENet(nn.Module):
    def __init__(self, data_schema, net_schema):
        super(DENet, self).__init__()

        self.__data_schema = data_schema
        self.__net_schema = net_schema

        self.dims = []
        for b in self.__net_schema["branches"]:
            inputSize = -1
            if self.__data_schema[b["key"]]["type"] == "spatial":
                res = self.__data_schema[b["key"]]["resolution"]
                inputSize = res["x"]+res["y"]
            else:
                inputSize = self.__data_schema[b["key"]]["dimension"]
            self.dims.append(inputSize)

        #######################################################################
        branches = [] # the layers of AE of each branch

        # Indexes (in branches[]) indicates where we should split in the middle of AE layers
        # so that we can get the intermediate results for regression
        # e.g. branches[i][idx]
        splitIDs = []
        codeIDs = [] # this ID of the real code layer when we talk about autoencoder
        outShapes = [] # the shape/size of AE output used in regression
        for b in self.__net_schema["branches"]:
            actf = None
            if b["activation"] == "ReLU":
                actf = nn.ReLU(inplace=True)
            elif b["activation"] == "LeakyReLU":
                actf = nn.LeakyReLU(negative_slope=0.1, inplace=True)

            if actf is None:
                print("Activation function not supported: {}".format(b["activation"]))
                exit(0)

            branch_layers = []

            inputSize = -1
            if self.__data_schema[b["key"]]["type"] == "spatial":
                res = self.__data_schema[b["key"]]["resolution"]
                inputSize = res["x"]+res["y"]
            else:
                inputSize = self.__data_schema[b["key"]]["dimension"]

            for i in range(len(b["layers"])):
                nType, nShape = b["layers"][i]
                if nType == "FC":
                    if i == 0:
                        # first layer
                        branch_layers.append(nn.Linear(inputSize, nShape))
                        branch_layers.append(actf)
                    else:
                        branch_layers.append(nn.Linear(b["layers"][i-1][1], nShape))
                        branch_layers.append(actf)

                    if 'batchNorm' in b and b['batchNorm']:
                        branch_layers.append(nn.BatchNorm1d(nShape))

                    if i == b["regLayerID"]:
                        splitIDs.append(len(branch_layers)-1)
                        outShapes.append(nShape)

                    if i == b["codeLayerID"]:
                        codeIDs.append(len(branch_layers)-2) # without the following activation layer
                else:
                    print("Network type not supported: {}".format(nType))
                    exit(0)

            # last layer of AE
            branch_layers.append(nn.Linear(b["layers"][-1][1], inputSize))
            # always use Sigmoid for last layer
            branch_layers.append(nn.Sigmoid())

            branches.append(branch_layers)

        # these are "half-way" encoders
        for i in range(len(branches)):
            setattr(self, "ec"+str(i), nn.Sequential(*branches[i][0:splitIDs[i]+1]))

        for i in range(len(branches)):
            setattr(self, "dc"+str(i), nn.Sequential(*branches[i][splitIDs[i]+1:]))

        # the real encoder when we talk about autoencoder
        # project the input into the smallest dimension e.g. input -> 2D
        for i in range(len(branches)):
            setattr(self, "project"+str(i), nn.Sequential(*branches[i][0:codeIDs[i]+1]))

        #######################################################################
        _reg = self.__net_schema["regressor"]
        reg_actf = None
        if _reg["activation"] == "ReLU":
            reg_actf = nn.ReLU(inplace=True)
        elif _reg["activation"] == "LeakyReLU":
            reg_actf = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        if reg_actf is None:
            print("Activation function not supported: {}".format(_reg["activation"]))
            exit(0)

        regLayers = []
        for i in range(len(_reg["layers"])):
            lType, lShape = _reg["layers"][i]
            if lType == "FC":
                if i == 0:
                    regLayers.append(nn.Linear(sum(outShapes), lShape))
                    regLayers.append(reg_actf)
                else:
                    regLayers.append(nn.Linear(_reg["layers"][i-1][1], lShape))
                    regLayers.append(reg_actf)
            else:
                print("Network type not supported: {}".format(nType))
                exit(0)

        # last layer of regressor
        regLayers.append(nn.Linear(_reg["layers"][-1][1], 1))
        # regLayers.append(nn.ReLU(inplace=True))
        self.regressor = nn.Sequential(*regLayers)

    def forward(self, x):
        codes = [] # intermediate AE output used in regressor, not necessarily the lowest dim layer
        startIdx = 0
        for i in range(len(self.dims)):
            f = getattr(self, "ec"+str(i))
            codes.append(f(x[:, startIdx : startIdx+self.dims[i]]))
            startIdx += self.dims[i]

        merged_code = torch.cat(codes, dim=1)
        out_regressor = self.regressor(merged_code)

        out_ae = []
        for i in range(len(self.dims)):
            f = getattr(self, "dc"+str(i))
            out_ae.append(f(codes[i]))

        return out_ae, out_regressor

    def project(self, x):
        codes = []
        startIdx = 0
        for i in range(len(self.dims)):
            f = getattr(self, "project"+str(i))
            codes.append(f(x[:, startIdx : startIdx+self.dims[i]]))
            startIdx += self.dims[i]
        return codes

    def schema(self):
        return {"data_schema": self.__data_schema, "net_schema": self.__net_schema}

    def logMeta(self, key, value):
        if "meta-data" not in self.__net_schema:
            self.__net_schema["meta-data"] = {}
        self.__net_schema["meta-data"][key] = str(value)

