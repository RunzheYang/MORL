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

'''
Code from
https://github.com/jaara/AI-blog/blob/master/SumTree.py
https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
Another thing is the initialization. Remember that before the learning itself, we fill
the memory using random agent. But this agent does not use any neural network, so how
could we estimate any error? We can use a fact that untrained neural network is likely
to return a value around zero for every input. In this case the error formula becomes very simple:
error = |Q(s, a) - T(S)| = |Q(s, a) - r - \gamma \tilde{Q}(s', argmax_a Q(s', a))| = | r |
The error in this case is simply the reward experienced in a given sample.
Indeed, the transitions where the agent experienced any reward intuitively seem to be very promising
'''

# s is the sampled value (from 0 to p_total)
# For known capacity, this sum tree data structure can be backed by an array?
#            0
#     1               2
#  3     4        5       6
# 7 8   9 10    11 12   13 14
# capacity = 8, tree size = 15,

import numpy

class SumTree:
    write = 0  # counter for capacity

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1) # indexes for sums - all leaves nodes plus the upper hierarchy thus 2 * capacity -1
        self.data = numpy.zeros(capacity, dtype=object)  # to store experience

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2  # index of parent

        self.tree[parent] += change  # adding the change that was made up the tree

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):  # get index in the tree of node where we wanted to get s
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        # search for the value
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):  # p_total - sum of all errors
        return self.tree[0]

    def add(self, p, experience):
        idx = self.write + self.capacity - 1

        self.data[self.write] = experience
        self.update(idx, p)  # p probability made from error

        self.write += 1  # if write is above capacity it starts from the first leaf again
        if self.write >= self.capacity:
            self.write = 0

    def getDataSize(self):
        return self.write
    
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p  # new error in the tree
        self._propagate(idx, change)  # propagating it back

    def get(self, s):
        idx = self._retrieve(0, s)  # index of a node
        dataIdx = idx - self.capacity + 1  # getting (7 - 8 + 1 = 0 - cause its first)

        return (idx, self.tree[idx], self.data[dataIdx])  # index, value of TD and experience
