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

import random
import numpy as np
import replay_abc
import PER.sum_tree


class ReplayPrioritised(replay_abc.ReplayABC):
    """
    stored as a tuple (s, a, r, s_1, terminal) in SumTree
    """

    def __init__(self, buffer_size, batch_size, random_seed=1234):
        """
        The right side of the deque contains the most recent experiences
        """
        self.tree = PER.sum_tree.SumTree(buffer_size)
        self.batch_size = batch_size
        self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev = None, None, None, None

        # p_i = (p + e)^a
        self.e = 0.00000001
        self.a = 0.6  # values suggested by authors
        self.beta = 0.4  # to 1 - values suggested by authors

        self.previous_index = None  # TODO
        self.prevQ_s_t_a_t_ = None
        random.seed(random_seed)

    def record(self, state, state_ori, action, reward, Q_s_t_a_t_, gamma_Q_s_tplu1_maxa_, uniform=False,
               terminal=False):

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

        if self.s_prev == None and self.s_ori_prev == None and self.a_prev == None and self.r_prev == None:
            self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev = state, state_ori, action, reward
            self.prevQ_s_t_a_t_ = Q_s_t_a_t_
            return
        else:
            if terminal == True:
                # add dialogue success reward to last added experience,
                # -1 for goodbye turn
                self.tree.data[self.tree.write - 1][3] += reward

                # change this experience to terminal
                self.tree.data[self.tree.write - 1][-1] = terminal

                # update the p (calculated from TD error) of node with idx in self.tree
                # idx: index in self.tree
                idx = self.tree.write + self.tree.capacity - 1 - 1  #last -1 to get back one position
                error = abs(self.prevQ_s_t_a_t_ - (self.r_prev + reward))  # reward final as q_s
                p = self._getPriority(error)
                self.tree.update(idx, p)

                self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev = None, None, None, None
                self.prevQ_s_t_a_t_ = None
            else:  # not terminal state
                if uniform == True:
                    error = 0.0
                else:
                    error = abs(self.prevQ_s_t_a_t_ - self.r_prev - gamma_Q_s_tplu1_maxa_)
                experience = [self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev, state, state_ori, terminal]
                self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev = state, state_ori, action, reward
                self.prevQ_s_t_a_t_ = Q_s_t_a_t_

                # add experience to tree buffer
                self.add(error, experience)

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
        probabilities = []

        # it takes floor so it doesnt take the last one into account
        segment = self.tree.total() / self.batch_size

        for ii in range(self.batch_size):
            a = segment * ii
            b = segment * (ii + 1)

            s = random.uniform(a, b)
            (idx, p, experience) = self.tree.get(s)
            if experience != 0:
                batch.append(experience)
                indexes.append(idx)
                probabilities.append(p)
                # ii += 1
            else:
                pass


        s_batch = np.array([_[0] for _ in batch])
        s_ori_batch = np.array([_[1] for _ in batch])
        a_batch = np.array([_[2] for _ in batch])
        r_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])
        s2_ori_batch = np.array([_[5] for _ in batch])
        t_batch = np.array([_[6] for _ in batch])

        # TODO - N means capacity - do we change it when it's not filled?
        weights = (np.array(probabilities) * self.tree.capacity) ** (-self.beta)
        w_max = max(weights)
        weights /= w_max

        return s_batch, s_ori_batch, a_batch, r_batch, s2_batch, s2_ori_batch, t_batch, indexes, None

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)
