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
Data structure for implementing prioritised experience replay, episodic level
"""
from collections import deque
import random
import numpy as np
from ..Policy import TerminalAction, TerminalState
import replay_abc
import PER.sum_tree
import sys


class ReplayPrioritisedEpisode(replay_abc.ReplayABC):
    """
    stored as a tuple (s, a, r, s_1, terminal) in SumTree
    """

    def __init__(self, buffer_size, batch_size, random_seed=1234):
        """
        The right side of the deque contains the most recent experiences
        """
        self.tree = PER.sum_tree.SumTree(buffer_size)
        self.batch_size = batch_size
        self.episode = []
        self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev, self.v_prev, self.distribution, self.mask = \
            None, None, None, None, None, None, None

        # p_i = (p + e)^a
        self.e = 0.00000001
        self.a = 0.6  # values suggested by authors
        self.beta = 0.4  # to 1 - values suggested by authors

        self.previous_index = None  # TODO
        random.seed(random_seed)

    def record(self, state, state_ori, action, reward, value, terminal=False, distribution=None, mask=None):
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

        if self.s_prev == None and self.s_ori_prev == None and self.a_prev == None and self.r_prev == None and \
                        self.distribution == None and self.mask == None:
            self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev, self.v_prev, self.distribution, self.mask = \
                state, state_ori, action, reward, value.tolist(), distribution, mask
            return
        else:
            self.episode.append(
                [self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev, self.v_prev, state, state_ori, terminal,
                 self.distribution, self.mask]
            )
            self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev, self.v_prev, self.distribution, self.mask = \
                state, state_ori, action, reward, value, distribution, mask

    def record_final_and_get_episode(self, state, state_ori, action, reward, value, terminal, distribution):
        # add dialogue success reward to last added experience,
        # -1 for goodbye turn
        self.episode[-1][3] += reward

        # change this experience to terminal
        self.episode[-1][7] = True

        episode_r = np.array([_[3] for _ in self.episode])
        episode_v = np.array([_[4] for _ in self.episode])

        return episode_r.tolist(), episode_v.tolist()

    def insertPriority(self, error):
        # update the p (calculated from TD error) of node with idx in self.tree
        # add experience to tree buffer
        self.add(error, self.episode)

        self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev, self.v_prev, self.distribution, self.mask = \
            None, None, None, None, None, None, None
        self.episode = []

    def size(self):
        return self.tree.getDataSize()  # todo change it

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, experience):
        p = self._getPriority(error)
        self.tree.add(p, experience)

    def sample_batch(self):  # batch size as n here
        batch = []
        indexes = []
        # self.batch_size = 1
        # it takes floor so it doesnt take the last one into account
        segment = self.tree.total() / self.batch_size

        # todo this is weird, sometimes the return is 0
        ii = 0
        # while ii < self.batch_size:
        for ii in range(self.batch_size):
            a = segment * ii
            b = segment * (ii + 1)
            # TODO test

            s = random.uniform(a, b)
            (idx, p, experience) = self.tree.get(s)
            if experience != 0:
                batch.append(experience)
                indexes.append(idx)
                # ii += 1
            else:
                pass
                # assert False
                # print ii, a, b, s, idx, p
                # break
        batch = np.array(batch)
        # print batch

        """
        print batch[0]
        print batch
        print 'len',len(batch)
        """

        s_batch = []
        s_ori_batch = []
        a_batch = []
        r_batch = []
        s2_batch = []
        s2_ori_batch = []
        t_batch = []
        v_batch = []
        d_batch = []
        m_batch = []
        return_distribution = False
        return_mask = False
        for epi in batch:
            tmp_s, tmp_s_ori, tmp_a, tmp_r, tmp_v, tmp_s2, tmp_s2_ori, tmp_t, tmp_d, tmp_m = \
                [], [], [], [], [], [], [], [], [], []
            for exp in epi:
                tmp_s.append(exp[0])
                tmp_s_ori.append(exp[1])
                tmp_a.append(exp[2])
                tmp_r.append(exp[3])
                tmp_v.append(exp[4])
                tmp_s2.append(exp[5])
                tmp_s2_ori.append(exp[6])
                tmp_t.append(exp[7])
                tmp_d.append(exp[8])
                tmp_m.append(exp[9])
                if exp[8] is not None:
                    return_distribution = True
                if exp[9] is not None:
                    return_mask = True

            s_batch.append(tmp_s)
            s_ori_batch.append(tmp_s_ori)
            a_batch.append(tmp_a)
            r_batch.append(tmp_r)
            s2_batch.append(tmp_s2)
            s2_ori_batch.append(tmp_s2_ori)
            t_batch.append(tmp_t)
            v_batch.append(tmp_v)
            d_batch.append(tmp_d)
            m_batch.append(tmp_m)

        print 'batch size', len(v_batch)
        """
        s_batch         = np.array([_[0] for _ in batch])
        s_ori_batch     = np.array([_[1] for _ in batch])
        a_batch         = np.array([_[2] for _ in batch])
        r_batch         = np.array([_[3] for _ in batch])
        s2_batch        = np.array([_[4] for _ in batch])
        s2_ori_batch    = np.array([_[5] for _ in batch])
        v_batch         = np.array([_[6] for _ in batch])
        t_batch         = np.array([_[7] for _ in batch])
        """
        ret = [s_batch, s_ori_batch, a_batch, r_batch, s2_batch, s2_ori_batch, t_batch, indexes, v_batch]
        if return_distribution:
            ret.append(d_batch)
        if return_mask:
            ret.append(m_batch)
        return ret

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)
