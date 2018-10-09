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

#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import rank_based


def test():
    conf = {'size': 50,
            'learn_start': 10,
            'partition_num': 5,
            'total_step': 100,
            'batch_size': 4}
    experience = rank_based.Experience(conf)

    # insert to experience
    print('test insert experience')
    for i in range(1, 51):
        # tuple, like(state_t, a, r, state_t_1, t)
        to_insert = (i, 1, 1, i, 1)
        experience.store(to_insert)
    print(experience.priority_queue)
    print(experience._experience[1])
    print(experience._experience[2])
    print('test replace')
    to_insert = (51, 1, 1, 51, 1)
    experience.store(to_insert)
    print(experience.priority_queue)
    print(experience._experience[1])
    print(experience._experience[2])

    # sample
    print('test sample')
    sample, w, e_id = experience.sample(51)
    print(sample)
    print(w)
    print(e_id)

    # update delta to priority
    print('test update delta')
    delta = [v for v in range(1, 5)]
    experience.update_priority(e_id, delta)
    print(experience.priority_queue)
    sample, w, e_id = experience.sample(51)
    print(sample)
    print(w)
    print(e_id)

    # rebalance
    print('test rebalance')
    experience.rebalance()
    print(experience.priority_queue)


def main():
    test()


if __name__ == '__main__':
    main()

