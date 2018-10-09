#########################################################
# CUED Python Statistical Spoken Dialogue System Software
#########################################################
#
# Copyright 2015-16  Cambridge University Engineering Department 
# Dialogue Systems Group
#
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
#########################################################

'''
DQNPolicy.py - deep Q network policy
==================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :class:`Policy`
    import :class:`utils.ContextLogger`

.. warning::
        Documentation not done.


************************

'''

import copy
import os
import sys
import json
import numpy as np
import cPickle as pickle
from itertools import product
import utils
from utils.Settings import config as cfg
from utils import ContextLogger, DiaAct, DialogueState

import ontology.FlatOntologyManager as FlatOnt
import tensorflow as tf
from policy.DRL.replay_buffer import ReplayBuffer
from policy.DRL.replay_prioritised import ReplayPrioritised
import policy.DRL.utils as drlutils
import policy.DRL.dqn as dqn
import policy.Policy
import policy.DQNPolicy
import policy.SummaryAction
from policy.Policy import TerminalAction, TerminalState
from policy.feudalRL.DIP_parametrisation import DIP_state, padded_state
from policy.feudalRL.feudalUtils import get_feudal_masks

logger = utils.ContextLogger.getLogger('')

class FeudalDQNPolicy(policy.DQNPolicy.DQNPolicy):
    '''Derived from :class:`DQNPolicy`
    '''

    def __init__(self, in_policy_file, out_policy_file, domainString='CamRestaurants', is_training=False,
                 action_names=None, slot=None):
        super(FeudalDQNPolicy, self).__init__(in_policy_file, out_policy_file, domainString, is_training)

        tf.reset_default_graph()

        self.domainString = domainString
        self.domainUtil = FlatOnt.FlatDomainOntology(self.domainString)
        self.in_policy_file = in_policy_file
        self.out_policy_file = out_policy_file
        self.is_training = is_training
        self.accum_belief = []
        self.slot = slot
        self.features = 'dip'
        if cfg.has_option('feudalpolicy', 'features'):
            self.features = cfg.get('feudalpolicy', 'features')
        self.actfreq_ds = False
        if cfg.has_option('feudalpolicy', 'actfreq_ds'):
            self.actfreq_ds = cfg.getboolean('feudalpolicy', 'actfreq_ds')

        self.domainUtil = FlatOnt.FlatDomainOntology(self.domainString)
        self.prev_state_check = None


        self.max_k = 5
        if cfg.has_option('dqnpolicy', 'max_k'):
            self.max_k = cfg.getint('dqnpolicy', 'max_k')

        self.capacity *= 4  # capacity for episode methods, multiply it to adjust to turn based methods

        # init session
        self.sess = tf.Session()
        with tf.device("/cpu:0"):

            np.random.seed(self.randomseed)
            tf.set_random_seed(self.randomseed)

            # initialise a replay buffer
            if self.replay_type == 'vanilla':
                self.episodes[self.domainString] = ReplayBuffer(self.capacity, self.minibatch_size, self.randomseed)
            elif self.replay_type == 'prioritized':
                self.episodes[self.domainString] = ReplayPrioritised(self.capacity, self.minibatch_size,
                                                                     self.randomseed)
            self.samplecount = 0
            self.episodecount = 0

            # construct the models
            self.state_dim = 89 # current DIP state dim
            self.summaryaction = policy.SummaryAction.SummaryAction(domainString)
            self.action_names = action_names
            self.action_dim = len(self.action_names)
            action_bound = len(self.action_names)
            self.stats = [0 for _ in range(self.action_dim)]

            if self.features == 'learned' or self.features == 'rnn':
                si_state_dim = 72
                if self.actfreq_ds:
                    if self.domainString == 'CamRestaurants':
                        si_state_dim += 9#16
                    elif self.domainString == 'SFRestaurants':
                        si_state_dim += 9#25
                    elif self.domainString == 'Laptops11':
                        si_state_dim += 9#40
                if self.domainString == 'CamRestaurants':
                    sd_state_dim = 158#94
                elif self.domainString == 'SFRestaurants':
                    sd_state_dim = 158
                elif self.domainString == 'Laptops11':
                    sd_state_dim = 158#13
                else:
                    logger.error('Domain {} not implemented in feudal-DQN yet') # just find out the size of sd_state_dim for the new domain
                self.sd_enc_size = 50
                self.si_enc_size = 25
                self.dropout_rate = 0.
                if cfg.has_option('feudalpolicy', 'sd_enc_size'):
                    self.sd_enc_size = cfg.getint('feudalpolicy', 'sd_enc_size')
                if cfg.has_option('feudalpolicy', 'si_enc_size'):
                    self.si_enc_size = cfg.getint('feudalpolicy', 'si_enc_size')
                if cfg.has_option('dqnpolicy', 'dropout_rate') and self.is_training:
                    self.dropout_rate = cfg.getfloat('feudalpolicy', 'dropout_rate')
                if cfg.has_option('dqnpolicy', 'dropout_rate') and self.is_training:
                    self.dropout_rate = cfg.getfloat('feudalpolicy', 'dropout_rate')

                self.state_dim = si_state_dim + sd_state_dim
                if self.features == 'learned':
                    self.dqn = dqn.NNFDeepQNetwork(self.sess, si_state_dim ,sd_state_dim, self.action_dim,
                                            self.learning_rate, self.tau, action_bound, self.minibatch_size,
                                            self.architecture, self.h1_size, self.h2_size, sd_enc_size=self.sd_enc_size,
                                               si_enc_size=self.si_enc_size, dropout_rate=self.dropout_rate)
                elif self.features == 'rnn':
                    self.dqn = dqn.RNNFDeepQNetwork(self.sess, si_state_dim, sd_state_dim, self.action_dim,
                                                   self.learning_rate, self.tau, action_bound, self.minibatch_size,
                                                   self.architecture, self.h1_size, self.h2_size,
                                                   sd_enc_size=self.sd_enc_size, si_enc_size=self.si_enc_size,
                                                   dropout_rate=self.dropout_rate, slot=self.slot)
            else: # self.features = 'dip'
                if self.actfreq_ds:
                    if self.domainString == 'CamRestaurants':
                        self.state_dim += 9#16
                    elif self.domainString == 'SFRestaurants':
                        self.state_dim += 9#25
                    elif self.domainString == 'Laptops11':
                        self.state_dim += 9#40
                self.dqn = dqn.DeepQNetwork(self.sess, self.state_dim, self.action_dim,
                                            self.learning_rate, self.tau, action_bound, self.minibatch_size,
                                            self.architecture, self.h1_size,
                                            self.h2_size, dropout_rate=self.dropout_rate)

            # when all models are defined, init all variables (this might to be sent to the main policy too)
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

            self.loadPolicy(self.in_policy_file)
            print 'loaded replay size: ', self.episodes[self.domainString].size()

            self.dqn.update_target_network()

    def record(self, reward, domainInControl=None, weight=None, state=None, action=None, exec_mask=None):
        if domainInControl is None:
            domainInControl = self.domainString
        if self.actToBeRecorded is None:
            self.actToBeRecorded = self.summaryAct

        if state is None:
            state = self.prevbelief
        if action is None:
            action = self.actToBeRecorded

        cState, cAction = state, action
        # normalising total return to -1~1
        reward /= 20.0

        cur_cState = np.vstack([np.expand_dims(x, 0) for x in [cState]])

        cur_action_q = self.dqn.predict(cur_cState)
        cur_target_q = self.dqn.predict_target(cur_cState)

        if exec_mask is not None:
            admissible = np.add(cur_target_q, np.array(exec_mask))
        else:
            admissible = cur_target_q
        Q_s_t_a_t_ = cur_action_q[0][cAction]
        gamma_Q_s_tplu1_maxa_ = self.gamma * np.max(admissible)

        if self.replay_type == 'vanilla':
            self.episodes[domainInControl].record(state=cState, \
                                                  state_ori=state, action=cAction, reward=reward)
        elif self.replay_type == 'prioritized':
            # heuristically assign 0.0 to Q_s_t_a_t_ and Q_s_tplu1_maxa_, doesn't matter as it is not used
            self.episodes[domainInControl].record(state=cState, \
                                                  state_ori=state, action=cAction, reward=reward, \
                                                  Q_s_t_a_t_=Q_s_t_a_t_,
                                                  gamma_Q_s_tplu1_maxa_=gamma_Q_s_tplu1_maxa_, uniform=False)
        self.actToBeRecorded = None
        self.samplecount += 1

    def finalizeRecord(self, reward, domainInControl=None):
        if domainInControl is None:
            domainInControl = self.domainString
        if self.episodes[domainInControl] is None:
            logger.warning("record attempted to be finalized for domain where nothing has been recorded before")
            return

        reward /= 20.0

        terminal_state, terminal_action = self.convertStateAction(TerminalState(), TerminalAction())

        if self.replay_type == 'vanilla':
            self.episodes[domainInControl].record(state=terminal_state, \
                                                  state_ori=TerminalState(), action=terminal_action, reward=reward,
                                                  terminal=True)
        elif self.replay_type == 'prioritized':
            self.episodes[domainInControl].record(state=terminal_state, \
                                                      state_ori=TerminalState(), action=terminal_action, reward=reward, \
                                                      Q_s_t_a_t_=0.0, gamma_Q_s_tplu1_maxa_=0.0, uniform=False,
                                                      terminal=True)
            print 'total TD', self.episodes[self.domainString].tree.total()

    def convertStateAction(self, state, action):
        '''

        '''
        if isinstance(state, TerminalState):
            return [0] * 89, action

        else:
            if self.features == 'learned' or self.features == 'rnn':
                dip_state = padded_state(state.domainStates[state.currentdomain], self.domainString)
            else:
                dip_state = DIP_state(state.domainStates[state.currentdomain], self.domainString)
            action_name = self.actions.action_names[action]
            act_slot = 'general'
            for slot in dip_state.slots:
                if slot in action_name:
                    act_slot = slot
            flat_belief = dip_state.get_beliefStateVec(act_slot)
            self.prev_state_check = flat_belief

            return flat_belief, action

    def nextAction(self, beliefstate):
        '''
        select next action

        :param beliefstate: already converted to dipstatevec of the specific slot (or general)
        :returns: (int) next summary action
        '''

        if self.exploration_type == 'e-greedy':
            # epsilon greedy
            if self.is_training and utils.Settings.random.rand() < self.epsilon:
                action_Q = np.random.rand(len(self.action_names))
            else:
                action_Q = self.dqn.predict(np.reshape(beliefstate, (1, len(beliefstate))))  # + (1. / (1. + i + j))
                # add current max Q to self.episode_ave_max_q
                self.episode_ave_max_q.append(np.max(action_Q))

        #return the Q vect, the action will be converted in the feudal policy
        return action_Q

    def train(self):
        '''
        call this function when the episode ends
        '''

        if not self.is_training:
            logger.info("Not in training mode")
            return
        else:
            logger.info("Update dqn policy parameters.")

        self.episodecount += 1
        logger.info("Sample Num so far: %s" % (self.samplecount))
        logger.info("Episode Num so far: %s" % (self.episodecount))

        if self.samplecount >= self.minibatch_size * 10 and self.episodecount % self.training_frequency == 0:
            logger.info('start training...')

            s_batch, s_ori_batch, a_batch, r_batch, s2_batch, s2_ori_batch, t_batch, idx_batch, _ = \
                self.episodes[self.domainString].sample_batch()

            s_batch = np.vstack([np.expand_dims(x, 0) for x in s_batch])
            s2_batch = np.vstack([np.expand_dims(x, 0) for x in s2_batch])

            a_batch_one_hot = np.eye(self.action_dim, self.action_dim)[a_batch]
            action_q = self.dqn.predict_dip(s2_batch, a_batch_one_hot)
            target_q = self.dqn.predict_target_dip(s2_batch, a_batch_one_hot)
            #print 'action Q and target Q:', action_q, target_q


            y_i = []
            for k in xrange(min(self.minibatch_size, self.episodes[self.domainString].size())):
                Q_bootstrap_label = 0
                if t_batch[k]:
                    Q_bootstrap_label = r_batch[k]
                else:
                    if self.q_update == 'single':
                        belief = s2_ori_batch[k]
                        #nonExec = self.getNonExecutable(belief.getDomainState(belief.currentdomain), a_batch[k])
                        #execMask = get_feudal_masks(nonExec,)
                        execMask = [0.0] * len(self.action_names) #TODO: find out how to compute the mask here, or save it when recording the state
                        execMask[-1] = -sys.maxint
                        action_Q = target_q[k]
                        admissible = np.add(action_Q, np.array(execMask))
                        # logger.info('action Q...')
                        # print admissible
                        Q_bootstrap_label = r_batch[k] + self.gamma * np.max(admissible)
                    elif self.q_update == 'double':
                        #execMask = self.summaryaction.getExecutableMask(s2_ori_batch[k], a_batch[k])
                        execMask = [0.0] * len(self.action_names)
                        execMask[-1] = -sys.maxint
                        action_Q = action_q[k]
                        value_Q = target_q[k]
                        admissible = np.add(action_Q, np.array(execMask))
                        Q_bootstrap_label = r_batch[k] + self.gamma * value_Q[np.argmax(admissible)]
                y_i.append(Q_bootstrap_label)

                if self.replay_type == 'prioritized':
                    # update the sum-tree
                    # update the TD error of the samples in the minibatch
                    currentQ_s_a_ = action_q[k][a_batch[k]]
                    error = abs(currentQ_s_a_ - Q_bootstrap_label)
                    self.episodes[self.domainString].update(idx_batch[k], error)

            # change index-based a_batch to one-hot-based a_batch
            #a_batch_one_hot = np.eye(self.action_dim, self.action_dim)[a_batch]

            # Update the critic given the targets
            reshaped_yi = np.vstack([np.expand_dims(x, 0) for x in y_i])

            predicted_q_value, _, currentLoss = self.dqn.train(s_batch, a_batch_one_hot, reshaped_yi)

            if self.episodecount % 1 == 0:
                # Update target networks
                self.dqn.update_target_network()

        self.savePolicyInc()


# END OF FILE
