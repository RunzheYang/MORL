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

""" 
Data structure for implementing experience replay, episode level
"""
from collections import deque
import random
import numpy as np
from ..Policy import TerminalAction, TerminalState
import replay_abc

class ReplayBufferEpisode():

    def __init__(self, buffer_size, batch_size, random_seed=1234):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.count = 0
        self.buffer = deque()
        self.episode = []
        self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev, self.v_prev, self.distribution_prev = None, None, None, None, None, None

        random.seed(random_seed)

    def record(self, state, state_ori, action, reward, value, distribution, terminal=False):
        """
        Record the experience:
            Turn #  User: state         System: action                  got reward
            Turn 1  User: Cheap         System: location?               -1
            Turn 2  User: North         System: inform(cheap, north)    -1
            Turn 3  User: Bye           System: inform(XXX) --> bye     -1
            Turn 4  User: Terminal_s    System: terminal_a              20

        As:
            Experience 1: (Cheap, location?, -1, North)
            Experience 2: (North, inform(cheap, north), -1+20, Bye)
        """

        if self.s_prev == None and self.s_ori_prev == None and self.a_prev == None and self.r_prev == None and self.distribution_prev == None:
            self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev, self.v_prev, self.distribution_prev = \
                            state, state_ori, action, reward, value.tolist(), distribution
            return
        else:
            if terminal == True:
                try:
                    #- 1  # add dialogue succes reward to last added experience , -1 for goodbye turn
                    self.episode[-1][3] += reward

                    # change this experience to terminal
                    self.episode[-1][-2] = terminal

                    # add episodic experience to buffer
                    if self.count < self.buffer_size:
                        self.buffer.append(self.episode)
                        self.count += 1
                    else:
                        self.buffer.popleft()
                        self.buffer.append(self.episode)

                    self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev, self.v_prev, self.distribution_prev = None, None, None, None, None, None
                    self.episode = []
                except:
                    self.episode = []
            else: # not terminal state
                self.episode.append(\
                        [self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev, state, state_ori, self.v_prev, terminal, self.distribution_prev])
                self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev, self.v_prev, self.distribution_prev = \
                                        state, state_ori, action, reward, value, distribution

    def size(self):
        return self.count

    def sample_batch(self):
        batch = []

        if self.count < self.batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, self.batch_size)

        batch = np.array(batch)

        s_batch         = []
        s_ori_batch     = []
        a_batch         = []
        r_batch         = []
        s2_batch        = []
        s2_ori_batch    = []
        v_batch         = []
        t_batch         = []
        mu_batch         = []

        for epi in batch:
            tmp_s, tmp_s_ori, tmp_a, tmp_r, tmp_s2, tmp_s2_ori, tmp_v, tmp_t, tmp_mu = \
                [], [], [], [], [], [], [], [], []
            for exp in epi:
                tmp_s.append(exp[0])
                tmp_s_ori.append(exp[1])
                tmp_a.append(exp[2])
                tmp_r.append(exp[3])
                tmp_s2.append(exp[4])
                tmp_s2_ori.append(exp[5])
                tmp_v.append(exp[6])
                tmp_t.append(exp[7])
                tmp_mu.append(exp[8])

            s_batch.append(tmp_s)
            s_ori_batch.append(tmp_s_ori)
            a_batch.append(tmp_a)
            r_batch.append(tmp_r)
            s2_batch.append(tmp_s2)
            s2_ori_batch.append(tmp_s2_ori)
            v_batch.append(tmp_v)
            t_batch.append(tmp_t)
            mu_batch.append(tmp_mu)

        """
        s_batch         = batch[:,:,0]
        s_ori_batch     = batch[:,:,1]
        a_batch         = batch[:,:,2]
        r_batch         = batch[:,:,3]
        s2_batch        = batch[:,:,4]
        s2_ori_batch    = batch[:,:,5]
        v_batch         = batch[:,:,6]
        t_batch         = batch[:,:,7]
        """

        return s_batch, s_ori_batch, a_batch, r_batch, s2_batch, s2_ori_batch, t_batch, None, v_batch, mu_batch

    """
    def sample_batch_vanilla_PER(self):
        ########
        # pick the samples for experience replay
        # based on prioritised sampling and random selection
        ########

        batch = random.sample(self.buffer, len(self.buffer))

        # determine pos and neg indices
        pos = []
        neg = []
        for r in range(len(batch)):
            if batch[r][2] > 0:
                #if Settings.random.rand() < self.prior_sample_prob:
                if random.random() < 0.8 and len(pos) < self.batch_size/2:
                    pos.append(batch[r])
            else:
                neg.append(batch[r])

        # first append all pos and then
        # fill up to self.minibatch_size with neg

        ExpRep = []
        ExpRep += pos

        random.shuffle(neg)
        neg = neg[-(self.batch_size-len(pos)):]

        ExpRep += neg

        s_batch         = np.array([_[0] for _ in batch])
        s_ori_batch     = np.array([_[1] for _ in batch])
        a_batch         = np.array([_[2] for _ in batch])
        r_batch         = np.array([_[3] for _ in batch])
        s2_batch        = np.array([_[4] for _ in batch])
        s2_ori_batch    = np.array([_[5] for _ in batch])
        t_batch         = np.array([_[6] for _ in batch])

        return s_batch, s_ori_batch, a_batch, r_batch, s2_batch, s2_ori_batch, t_batch, None, None
    """

    def clear(self):
        self.deque.clear()
        self.count = 0
        self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev, self.v_prev = None, None, None, None, None
        self.episode = []
