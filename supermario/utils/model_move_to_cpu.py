import torch

import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

model = torch.load("{}{}.pkl".format("saved/", "m.naive_cnn_n.naive_gpu_3_tmp"))
torch.save(model.cpu(), "{}{}.pkl".format("saved/cpu/", "m.naive_cnn_n.naive_gpu_3_tmp"))