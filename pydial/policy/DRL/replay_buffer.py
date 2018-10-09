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
Data structure for implementing experience replay

Author: Patrick Emami
"""
from collections import deque
import random
import numpy as np
from ..Policy import TerminalAction, TerminalState
import replay_abc

class ReplayBuffer(replay_abc.ReplayABC):

    def __init__(self, buffer_size, batch_size, random_seed=1234):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.count = 0
        self.buffer = deque()
        self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev = None, None, None, None

        random.seed(random_seed)

    def record(self, state, state_ori, action, reward, terminal=False):
        """
        Record the experience:
            Turn #  User: state         System: action                  got reward
            Turn 1  User: Cheap         System: location?               -1
            Turn 2  User: North         System: inform(cheap, north)    -1    
            Turn 3  User: Bye           System: inform(XXX) --> bye     -1
            Turn 4  User: Terminal_s    System: terminal_a              20

        As:
            Experience 1: (Cheap, location?, -1, North)
            Experience 2: (North, inform(cheap, north), -1, Bye)
            Experience 3: (Bye, bye, -1+20, Terminal_s)

            OLD -- tentative reward --
            OLD Experience 1: (Cheap, location?, -1, North)
            OLD Experience 2: (North, inform(cheap, north), -1+20, Bye)

        """

        if self.s_prev is None and self.s_ori_prev is None and self.a_prev is None and self.r_prev is None:
            self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev = state, state_ori, action, reward
            return
        else:
            #terminal = False
            #if isinstance(state,TerminalState):
            if terminal == True:
                # if current state is terminal (dummy), add dialogue success reward to r_prev
                #terminal = True
                #experience = (self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev+reward, state, state_ori, terminal)

                #self.buffer.pop()

                #action = 13 # bye action
                #self.buffer[-1][1] = action # action here is 13
                
                self.buffer[-1][3] += reward #- 1  # add dialogue succes reward to last added experience , -1 for goodbye turn
                self.buffer[-1][-1] = terminal # change this experience to terminal

                #print 'terminal transition...', self.buffer[-1]

                #experience = self.buffer.popright()
                #modifiedExp = list(experience[:2] + [experience[2]+reward-1] + experience[3:-1] + [terminal])

                self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev = None, None, None, None
            else: # not terminal state
                experience = [self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev, state, state_ori, terminal]
                self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev = state, state_ori, action, reward

                # add experience to buffer
                if self.count < self.buffer_size:
                    self.buffer.append(experience)
                    self.count += 1
                else:
                    self.buffer.popleft()
                    self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self):
        batch = []

        """
        s_batch  = np.array([_[0] for _ in self.buffer])
        a_batch  = np.array([_[1] for _ in self.buffer])
        r_batch  = np.array([_[2] for _ in self.buffer])
        s2_batch = np.array([_[3] for _ in self.buffer])
        t_batch  = np.array([_[4] for _ in self.buffer])
        print s_batch
        print a_batch
        print r_batch
        print s2_batch
        print t_batch
        """
        if self.count < self.batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, self.batch_size)

        s_batch         = np.array([_[0] for _ in batch])
        s_ori_batch     = np.array([_[1] for _ in batch])
        a_batch         = np.array([_[2] for _ in batch])
        r_batch         = np.array([_[3] for _ in batch])
        s2_batch        = np.array([_[4] for _ in batch])
        s2_ori_batch    = np.array([_[5] for _ in batch])
        t_batch         = np.array([_[6] for _ in batch])

        return s_batch, s_ori_batch, a_batch, r_batch, s2_batch, s2_ori_batch, t_batch, None, None

    def sample_batch_vanilla_PER(self):
        ########
        # pick the samples for experience replay 
        # based on prioritised sampling and random selection
        ########

        batch = random.sample(self.buffer, len(self.buffer))

        """
        print 'length of batch', len(batch)
        print self.batch_size
        print self.batch_size/2
        print self.batch_size-self.batch_size/2
        """

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


    def clear(self):
        self.deque.clear()
        self.count = 0
        self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev = None, None, None, None

