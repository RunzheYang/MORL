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

from collections import deque
import random

class ReplayABC(object):

    def __init__(self, buffer_size, batch_size, random_seed=1234):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.count = 0
        self.buffer = deque()
        self.s_pprev, self.s_ori_pprev, self.a_pprev, self.r_pprev = None, None, None, None
        self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev = None, None, None, None
        random.seed(random_seed)

    def record(self, state, state_ori, action, reward, terminal=False):
        pass
    
    def size(self):
        return self.count

    def sample_batch(self):
        pass

    def clear(self):
        self.deque.clear()
        self.count = 0
        self.s_pprev, self.s_ori_pprev, self.a_pprev, self.r_pprev = None, None, None, None
        self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev = None, None, None, None
