###############################################################################
# PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015 - 2017
# Cambridge University Engineering Department Dialogue Systems Group
#
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###############################################################################

import os
import sys
import time
import math

import numpy as np

def readFeatureFile(filename, nn_type = 'dnn'):
    content = []
    file = open(filename,'r')
    for line in file:
        tmp = line.strip().split(',')
        if tmp[-1] == '':
            del tmp[-1]
        if 'rnn' in nn_type or 'encdec' in nn_type:
            content.append([[float(i) for i in tmp]])
        else:
            content.append([float(i) for i in tmp])
    return content

def build_oneHot_targets(return_value):
    bins = range(0,13)  # 13 included, totally 14
    target = []
    for i in bins:
        if i == return_value:
            target.append(1)
        else:
            target.append(0)

    return target

def readTargetFile(filename):
    content = []
    file = open(filename,'r')
    for line in file:
        tmp = line.strip()
        #content.append(build_oneHot_targets(float(tmp)))
        #content.append([(float(tmp))])
        content.append((float(tmp)))
    return content

def readMultiTargetFile(filename):
    content = []
    file = open(filename,'r')
    for line in file:
        tmp = line.strip().split(',')
        #content.append(build_oneHot_targets(float(tmp)))
        tmp = [float(i) for i in tmp]
        #if len(tmp) == 1:
        #    tmp = tmp[0]
        content.append(tmp)
    return content

def readLineFile(filename):
    content = []
    file = open(filename,'r')
    for line in file:
        tmp = line.strip()
        content.append((int(tmp)))
    return content


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, goals, batchsizeList, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    #for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
    cntIdx = 0
    for idx in batchsizeList:
        if shuffle:
            excerpt = indices[cntIdx,cntIdx+idx]
        else:
            excerpt = slice(cntIdx,cntIdx+idx)
        cntIdx += idx
        yield inputs[excerpt], targets[excerpt], goals[idx]


