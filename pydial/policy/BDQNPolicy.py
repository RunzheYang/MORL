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
import json
import numpy as np
import cPickle as pickle
import random
import utils
from utils.Settings import config as cfg
from utils import ContextLogger, DiaAct

import ontology.FlatOntologyManager as FlatOnt
import tensorflow as tf
from DRL.replay_buffer import ReplayBuffer
from DRL.replay_prioritised import ReplayPrioritised
import DRL.utils as drlutils
import DRL.bdqn as bbqn
import Policy
import SummaryAction
from Policy import TerminalAction, TerminalState

logger = utils.ContextLogger.getLogger('')

# --- for flattening the belief --- #
def flatten_belief(belief, domainUtil, merge=False):
    belief = belief.getDomainState(domainUtil.domainString)
    if isinstance(belief, TerminalState):
        if domainUtil.domainString == 'CamRestaurants':
            return [0] * 268
        elif domainUtil.domainString == 'CamHotels':
            return [0] * 111
        elif domainUtil.domainString == 'SFRestaurants':
            return [0] * 633
        elif domainUtil.domainString == 'SFHotels':
            return [0] * 438
        elif domainUtil.domainString == 'Laptops11':
            return [0] * 257
        elif domainUtil.domainString == 'TV':
            return [0] * 188

    policyfeatures = ['full', 'method', 'discourseAct', 'requested', \
                      'lastActionInformNone', 'offerHappened', 'inform_info']

    flat_belief = []
    for feat in policyfeatures:
        add_feature = []
        if feat == 'full':
            # for slot in self.sorted_slots:
            for slot in domainUtil.ontology['informable']:
                for value in domainUtil.ontology['informable'][slot]:  # + ['**NONE**']:
                    add_feature.append(belief['beliefs'][slot][value])

                # pfb30 11.03.2017
                try:
                    add_feature.append(belief['beliefs'][slot]['**NONE**'])
                except:
                    add_feature.append(0.)  # for NONE
                try:
                    add_feature.append(belief['beliefs'][slot]['dontcare'])
                except:
                    add_feature.append(0.)  # for dontcare

        elif feat == 'method':
            add_feature = [belief['beliefs']['method'][method] for method in domainUtil.ontology['method']]
        elif feat == 'discourseAct':
            add_feature = [belief['beliefs']['discourseAct'][discourseAct]
                           for discourseAct in domainUtil.ontology['discourseAct']]
        elif feat == 'requested':
            add_feature = [belief['beliefs']['requested'][slot] \
                           for slot in domainUtil.ontology['requestable']]
        elif feat == 'lastActionInformNone':
            add_feature.append(float(belief['features']['lastActionInformNone']))
        elif feat == 'offerHappened':
            add_feature.append(float(belief['features']['offerHappened']))
        elif feat == 'inform_info':
            add_feature += belief['features']['inform_info']
        else:
            logger.error('Invalid feature name in config: ' + feat)

        flat_belief += add_feature

    return flat_belief


class BDQNPolicy(Policy.Policy):
    '''Derived from :class:`Policy`
    '''

    def __init__(self, in_policy_file, out_policy_file, domainString='CamRestaurants', is_training=False):
        super(BDQNPolicy, self).__init__(domainString, is_training)

        tf.reset_default_graph()

        self.domainString = domainString
        self.domainUtil = FlatOnt.FlatDomainOntology(self.domainString)
        self.in_policy_file = in_policy_file
        self.out_policy_file = out_policy_file
        self.is_training = is_training
        self.accum_belief = []

        self.domainUtil = FlatOnt.FlatDomainOntology(self.domainString)
        self.prev_state_check = None

        # parameter settings
        self.n_in = 260
        if cfg.has_option('dqnpolicy', 'n_in'):
            self.n_in = cfg.getint('dqnpolicy', 'n_in')

        self.actor_lr = 0.0001
        if cfg.has_option('dqnpolicy', 'actor_lr'):
            self.actor_lr = cfg.getfloat('dqnpolicy', 'actor_lr')

        self.critic_lr = 0.001
        if cfg.has_option('dqnpolicy', 'critic_lr'):
            self.critic_lr = cfg.getfloat('dqnpolicy', 'critic_lr')

        self.tau = 0.001
        if cfg.has_option('dqnpolicy', 'tau'):
            self.tau = cfg.getfloat('dqnpolicy', 'tau')

        self.randomseed = 1234
        if cfg.has_option('GENERAL', 'seed'):
            self.randomseed = cfg.getint('GENERAL', 'seed')

        self.gamma = 1.0
        if cfg.has_option('dqnpolicy', 'gamma'):
            self.gamma = cfg.getfloat('dqnpolicy', 'gamma')

        self.regularisation = 'l2'
        if cfg.has_option('dqnpolicy', 'regularisation'):
            self.regularisation = cfg.get('dqnpolicy', 'regulariser')

        self.learning_rate = 0.001  # ct506 #0.001
        if cfg.has_option('dqnpolicy', 'learning_rate'):
            self.learning_rate = cfg.getfloat('dqnpolicy', 'learning_rate')

        self.exploration_type = 'e-greedy'  # Boltzman
        if cfg.has_option('dqnpolicy', 'exploration_type'):
            self.exploration_type = cfg.get('dqnpolicy', 'exploration_type')

        self.episodeNum = 1000
        if cfg.has_option('dqnpolicy', 'episodeNum'):
            self.episodeNum = cfg.getfloat('dqnpolicy', 'episodeNum')

        self.maxiter = 5000
        if cfg.has_option('dqnpolicy', 'maxiter'):
            self.maxiter = cfg.getfloat('dqnpolicy', 'maxiter')

        self.epsilon = 1
        if cfg.has_option('dqnpolicy', 'epsilon'):
            self.epsilon = cfg.getfloat('dqnpolicy', 'epsilon')

        self.epsilon_start = 1
        if cfg.has_option('dqnpolicy', 'epsilon_start'):
            self.epsilon_start = cfg.getfloat('dqnpolicy', 'epsilon_start')

        self.epsilon_end = 1
        if cfg.has_option('dqnpolicy', 'epsilon_end'):
            self.epsilon_end = cfg.getfloat('dqnpolicy', 'epsilon_end')

        self.priorProbStart = 1.0
        if cfg.has_option('dqnpolicy', 'prior_sample_prob_start'):
            self.priorProbStart = cfg.getfloat('dqnpolicy', 'prior_sample_prob_start')

        self.priorProbEnd = 0.1
        if cfg.has_option('dqnpolicy', 'prior_sample_prob_end'):
            self.priorProbEnd = cfg.getfloat('dqnpolicy', 'prior_sample_prob_end')

        self.policyfeatures = []
        if cfg.has_option('dqnpolicy', 'features'):
            logger.info('Features: ' + str(cfg.get('dqnpolicy', 'features')))
            self.policyfeatures = json.loads(cfg.get('dqnpolicy', 'features'))

        self.max_k = 5
        if cfg.has_option('dqnpolicy', 'max_k'):
            self.max_k = cfg.getint('dqnpolicy', 'max_k')

        self.learning_algorithm = 'drl'
        if cfg.has_option('dqnpolicy', 'learning_algorithm'):
            self.learning_algorithm = cfg.get('dqnpolicy', 'learning_algorithm')
            logger.info('Learning algorithm: ' + self.learning_algorithm)

        self.minibatch_size = 32
        if cfg.has_option('dqnpolicy', 'minibatch_size'):
            self.minibatch_size = cfg.getint('dqnpolicy', 'minibatch_size')

        self.capacity = 1000  # max(self.minibatch_size, 2000)
        if cfg.has_option('dqnpolicy', 'capacity'):
            self.capacity = max(cfg.getint('dqnpolicy', 'capacity'), 2000)

        self.replay_type = 'vanilla'
        if cfg.has_option('dqnpolicy', 'replay_type'):
            self.replay_type = cfg.get('dqnpolicy', 'replay_type')

        self.architecture = 'vanilla'
        if cfg.has_option('dqnpolicy', 'architecture'):
            self.architecture = cfg.get('dqnpolicy', 'architecture')

        self.q_update = 'single'
        if cfg.has_option('dqnpolicy', 'q_update'):
            self.q_update = cfg.get('dqnpolicy', 'q_update')

        self.h1_size = 130
        if cfg.has_option('dqnpolicy', 'h1_size'):
            self.h1_size = cfg.getint('dqnpolicy', 'h1_size')

        self.h2_size = 130
        if cfg.has_option('dqnpolicy', 'h2_size'):
            self.h2_size = cfg.getint('dqnpolicy', 'h2_size')

        self.save_step = 200
        if cfg.has_option('policy', 'save_step'):
            self.save_step = cfg.getint('policy', 'save_step')

        # BDQN parameteres
        self.n_samples = 1
        if cfg.has_option('dqnpolicy', 'n_samples'):
            self.n_samples = cfg.getint('dqnpolicy', 'n_samples')

        sigma_prior = 1.5  # np.array(-3.0, dtype=np.float32)
        if cfg.has_option('dqnpolicy', 'sigma_prior'):
            sigma_prior = cfg.getfloat('dqnpolicy', 'sigma_prior')
        self.sigma_prior = tf.exp(sigma_prior)  # np.exp(np.array(sigma_prior, dtype=np.float32))

        self.stddev_var_mu = 0.01
        if cfg.has_option('dqnpolicy', 'stddev_var_mu'):
            self.stddev_var_mu = cfg.getfloat('dqnpolicy', 'stddev_var_mu')

        self.stddev_var_logsigma = 0.01
        if cfg.has_option('dqnpolicy', 'stddev_var_logsigma'):
            self.stddev_var_logsigma = cfg.getfloat('dqnpolicy', 'stddev_var_logsigma')

        self.mean_log_sigma = 0.000001
        if cfg.has_option('dqnpolicy', 'mean_log_sigma'):
            self.mean_log_sigma = cfg.getfloat('dqnpolicy', 'mean_log_sigma')

        self.n_batches = 1000.0
        if cfg.has_option('dqnpolicy', 'n_batches'):
            self.n_batches = cfg.getfloat('dqnpolicy', 'n_batches')

        self.importance_sampling = False
        if cfg.has_option('dqnpolicy', 'importance_sampling'):
            self.importance_sampling = cfg.getboolean('dqnpolicy', 'importance_sampling')

        self.alpha = 0.85
        if cfg.has_option('dqnpolicy', 'alpha'):
            self.alpha = cfg.getfloat('dqnpolicy', 'alpha')

        self.alpha_divergence = False
        if cfg.has_option('dqnpolicy', 'alpha_divergence'):
            self.alpha_divergence = cfg.getboolean('dqnpolicy', 'alpha_divergence')

        self.sigma_eps = 0.01
        if cfg.has_option('dqnpolicy', 'sigma_eps'):
            self.sigma_eps = cfg.getfloat('dqnpolicy', 'sigma_eps')

        self.training_frequency = 2
        if cfg.has_option('dqnpolicy', 'training_frequency'):
            self.training_frequency = cfg.getint('dqnpolicy', 'training_frequency')

        # domain specific parameter settings (overrides general policy parameter settings)
        if cfg.has_option('dqnpolicy_' + domainString, 'n_in'):
            self.n_in = cfg.getint('dqnpolicy_' + domainString, 'n_in')

        if cfg.has_option('dqnpolicy_' + domainString, 'actor_lr'):
            self.actor_lr = cfg.getfloat('dqnpolicy_' + domainString, 'actor_lr')

        if cfg.has_option('dqnpolicy_' + domainString, 'critic_lr'):
            self.critic_lr = cfg.getfloat('dqnpolicy_' + domainString, 'critic_lr')

        if cfg.has_option('dqnpolicy_' + domainString, 'tau'):
            self.tau = cfg.getfloat('dqnpolicy_' + domainString, 'tau')

        if cfg.has_option('dqnpolicy_' + domainString, 'gamma'):
            self.gamma = cfg.getfloat('dqnpolicy_' + domainString, 'gamma')

        if cfg.has_option('dqnpolicy_' + domainString, 'regularisation'):
            self.regularisation = cfg.get('dqnpolicy_' + domainString, 'regulariser')

        if cfg.has_option('dqnpolicy_' + domainString, 'learning_rate'):
            self.learning_rate = cfg.getfloat('dqnpolicy_' + domainString, 'learning_rate')

        if cfg.has_option('dqnpolicy_' + domainString, 'exploration_type'):
            self.exploration_type = cfg.get('dqnpolicy_' + domainString, 'exploration_type')

        if cfg.has_option('dqnpolicy_' + domainString, 'episodeNum'):
            self.episodeNum = cfg.getfloat('dqnpolicy_' + domainString, 'episodeNum')

        if cfg.has_option('dqnpolicy_' + domainString, 'maxiter'):
            self.maxiter = cfg.getfloat('dqnpolicy_' + domainString, 'maxiter')

        if cfg.has_option('dqnpolicy_' + domainString, 'epsilon'):
            self.epsilon = cfg.getfloat('dqnpolicy_' + domainString, 'epsilon')

        if cfg.has_option('dqnpolicy_' + domainString, 'epsilon_start'):
            self.epsilon_start = cfg.getfloat('dqnpolicy_' + domainString, 'epsilon_start')

        if cfg.has_option('dqnpolicy_' + domainString, 'epsilon_end'):
            self.epsilon_end = cfg.getfloat('dqnpolicy_' + domainString, 'epsilon_end')

        if cfg.has_option('dqnpolicy_' + domainString, 'prior_sample_prob_start'):
            self.priorProbStart = cfg.getfloat('dqnpolicy_' + domainString, 'prior_sample_prob_start')

        if cfg.has_option('dqnpolicy_' + domainString, 'prior_sample_prob_end'):
            self.priorProbEnd = cfg.getfloat('dqnpolicy_' + domainString, 'prior_sample_prob_end')

        if cfg.has_option('dqnpolicy_' + domainString, 'features'):
            logger.info('Features: ' + str(cfg.get('dqnpolicy_' + domainString, 'features')))
            self.policyfeatures = json.loads(cfg.get('dqnpolicy_' + domainString, 'features'))

        if cfg.has_option('dqnpolicy_' + domainString, 'max_k'):
            self.max_k = cfg.getint('dqnpolicy_' + domainString, 'max_k')

        if cfg.has_option('dqnpolicy_' + domainString, 'learning_algorithm'):
            self.learning_algorithm = cfg.get('dqnpolicy_' + domainString, 'learning_algorithm')
            logger.info('Learning algorithm: ' + self.learning_algorithm)

        if cfg.has_option('dqnpolicy_' + domainString, 'minibatch_size'):
            self.minibatch_size = cfg.getint('dqnpolicy_' + domainString, 'minibatch_size')

        if cfg.has_option('dqnpolicy_' + domainString, 'capacity'):
            self.capacity = cfg.getint('dqnpolicy_' + domainString, 'capacity')

        if cfg.has_option('dqnpolicy_' + domainString, 'replay_type'):
            self.replay_type = cfg.get('dqnpolicy_' + domainString, 'replay_type')

        if cfg.has_option('dqnpolicy_' + domainString, 'architecture'):
            self.architecture = cfg.get('dqnpolicy_' + domainString, 'architecture')

        if cfg.has_option('dqnpolicy_' + domainString, 'q_update'):
            self.q_update = cfg.get('dqnpolicy_' + domainString, 'q_update')

        if cfg.has_option('dqnpolicy_' + domainString, 'h1_size'):
            self.h1_size = cfg.getint('dqnpolicy_' + domainString, 'h1_size')

        if cfg.has_option('dqnpolicy_' + domainString, 'h2_size'):
            self.h2_size = cfg.getint('dqnpolicy_' + domainString, 'h2_size')

        if cfg.has_option('policy_' + domainString, 'save_step'):
            self.save_step = cfg.getint('policy_' + domainString, 'save_step')

        # BDQN parameteres
        if cfg.has_option('dqnpolicy_' + domainString, 'n_samples'):
            self.n_samples = cfg.getint('dqnpolicy_' + domainString, 'n_samples')

        if cfg.has_option('dqnpolicy_' + domainString, 'sigma_prior'):
            sigma_prior = cfg.getfloat('dqnpolicy_' + domainString, 'sigma_prior')
        self.sigma_prior = tf.exp(sigma_prior)  # np.exp(np.array(sigma_prior, dtype=np.float32))

        if cfg.has_option('dqnpolicy_' + domainString, 'stddev_var_mu'):
            self.stddev_var_mu = cfg.getfloat('dqnpolicy_' + domainString, 'stddev_var_mu')

        if cfg.has_option('dqnpolicy_' + domainString, 'stddev_var_logsigma'):
            self.stddev_var_logsigma = cfg.getfloat('dqnpolicy_' + domainString, 'stddev_var_logsigma')

        if cfg.has_option('dqnpolicy_' + domainString, 'mean_log_sigma'):
            self.mean_log_sigma = cfg.getfloat('dqnpolicy_' + domainString, 'mean_log_sigma')

        if cfg.has_option('dqnpolicy_' + domainString, 'n_batches'):
            self.n_batches = cfg.getfloat('dqnpolicy_' + domainString, 'n_batches')

        if cfg.has_option('dqnpolicy_' + domainString, 'importance_sampling'):
            self.importance_sampling = cfg.getboolean('dqnpolicy_' + domainString, 'importance_sampling')

        if cfg.has_option('dqnpolicy_' + domainString, 'alpha'):
            self.alpha = cfg.getfloat('dqnpolicy_' + domainString, 'alpha')

        if cfg.has_option('dqnpolicy_' + domainString, 'alpha_divergence'):
            self.alpha_divergence = cfg.getboolean('dqnpolicy_' + domainString, 'alpha_divergence')

        if cfg.has_option('dqnpolicy_' + domainString, 'sigma_eps'):
            self.sigma_eps = cfg.getfloat('dqnpolicy_' + domainString, 'sigma_eps')

        if cfg.has_option('dqnpolicy_' + domainString, 'training_frequency'):
            self.training_frequency = cfg.getint('dqnpolicy_' + domainString, 'training_frequency')

        print 'ct506', 'sigma_eps', self.sigma_eps, 'lr', self.learning_rate, 'm', self.n_batches
        self.episode_ave_max_q = []

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # init session
        self.sess = tf.Session()
        with tf.device("/cpu:0"):

            np.random.seed(self.randomseed)
            tf.set_random_seed(self.randomseed)

            # initialise an replay buffer
            if self.replay_type == 'vanilla':
                self.episodes[self.domainString] = ReplayBuffer(self.capacity, self.minibatch_size, self.randomseed)
            elif self.replay_type == 'prioritized':
                self.episodes[self.domainString] = ReplayPrioritised(self.capacity, self.minibatch_size,
                                                                     self.randomseed)
            # replay_buffer = ReplayBuffer(self.capacity, self.randomseed)
            # self.episodes = []
            self.samplecount = 0
            self.episodecount = 0

            # construct the models
            self.state_dim = self.n_in
            self.summaryaction = SummaryAction.SummaryAction(domainString)
            self.action_dim = len(self.summaryaction.action_names)
            action_bound = len(self.summaryaction.action_names)
            self.stats = [0 for _ in range(self.action_dim)]
            self.stdVar = []
            self.meanVar = []
            self.stdMean = []
            self.meanMean = []
            self.td_error = []
            self.td_errorVar = []

            self.bbqn = bbqn.DeepQNetwork(self.sess, self.state_dim, self.action_dim, self.learning_rate, self.tau, action_bound, self.architecture, self.h1_size, self.h2_size, self.n_samples, self.minibatch_size, self.sigma_prior, self.n_batches, self.stddev_var_mu,  self.stddev_var_logsigma, self.mean_log_sigma, self.importance_sampling, self.alpha_divergence, self.alpha, self.sigma_eps)

            # when all models are defined, init all variables
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

            self.loadPolicy(self.in_policy_file)
            print 'loaded replay size: ', self.episodes[self.domainString].size()

            self.bbqn.update_target_network()

    # def record() has been handled...

    def act_on(self, state, hyps=None):
        if self.lastSystemAction is None and self.startwithhello:
            systemAct, nextaIdex = 'hello()', -1
        else:
            systemAct, nextaIdex = self.nextAction(state)
        self.lastSystemAction = systemAct
        self.summaryAct = nextaIdex
        self.prevbelief = state

        systemAct = DiaAct.DiaAct(systemAct)
        return systemAct

    def record(self, reward, domainInControl=None, weight=None, state=None, action=None):
        if domainInControl is None:
            domainInControl = self.domainString
        if self.actToBeRecorded is None:
            # self.actToBeRecorded = self.lastSystemAction
            self.actToBeRecorded = self.summaryAct

        if state is None:
            state = self.prevbelief
        if action is None:
            action = self.actToBeRecorded

        cState, cAction = self.convertStateAction(state, action)

        # normalising total return to -1~1
        reward /= 20.0

        cur_cState = np.vstack([np.expand_dims(x, 0) for x in [cState]])
        cur_action_q = self.bbqn.predict(cur_cState)
        cur_target_q = self.bbqn.predict_target(cur_cState)
        execMask = self.summaryaction.getExecutableMask(state, cAction)

        if self.q_update == 'single':
            admissible = np.add(cur_target_q, np.array(execMask))
            Q_s_t_a_t_ = cur_action_q[0][cAction]
            gamma_Q_s_tplu1_maxa_ = self.gamma * np.max(admissible)
        elif self.q_update == 'double':
            admissible = np.add(cur_action_q, np.array(execMask))
            Q_s_t_a_t_ = cur_action_q[0][cAction]
            target_value_Q = cur_target_q[0]
            gamma_Q_s_tplu1_maxa_ = self.gamma * target_value_Q[np.argmax(admissible)]

        #print 'Q_s_t_a_t_', Q_s_t_a_t_
        #print 'gamma_Q_s_tplu1_maxa_', gamma_Q_s_tplu1_maxa_
        """
        s_batch = np.vstack([np.expand_dims(x, 0) for x in s_batch])
        s2_batch = np.vstack([np.expand_dims(x, 0) for x in s2_batch])
        #target_q = self.bbqn.predict_target_with_action_maxQ(s2_batch)
        action_q = self.bbqn.predict(s2_batch)
        target_q = self.bbqn.predict_target(s2_batch)

        y_i = []
        for k in xrange(min(self.minibatch_size, self.episodes[self.domainString].size())):
            Q_bootstrap_label = 0
            if t_batch[k]:
                Q_bootstrap_label = r_batch[k]
            else:
                if self.q_update == 'single':
                    execMask = self.summaryaction.getExecutableMask(s2_ori_batch[k], a_batch[k])
                    action_Q = target_q[k]
                    admissible = np.add(action_Q, np.array(execMask))
                    #logger.info('action Q...')
                    #print admissible
                    Q_bootstrap_label = r_batch[k] + self.gamma * np.max(admissible)
                elif self.q_update == 'double':
                    execMask = self.summaryaction.getExecutableMask(s2_ori_batch[k], a_batch[k])
                    action_Q = action_q[k]
                    value_Q = target_q[k]
                    admissible = np.add(action_Q, np.array(execMask))
                    Q_bootstrap_label = r_batch[k] + self.gamma * value_Q[np.argmax(admissible)]
            y_i.append(Q_bootstrap_label)
        """

        if self.replay_type == 'vanilla':
            self.episodes[domainInControl].record(state=cState, \
                                                  state_ori=state, action=cAction, reward=reward)
        elif self.replay_type == 'prioritized':
            self.episodes[domainInControl].record(state=cState, \
                                                      state_ori=state, action=cAction, reward=reward, \
                                                      Q_s_t_a_t_=Q_s_t_a_t_,
                                                      gamma_Q_s_tplu1_maxa_=gamma_Q_s_tplu1_maxa_, uniform=False)


        self.actToBeRecorded = None
        self.samplecount += 1
        return

    def finalizeRecord(self, reward, domainInControl=None):
        if domainInControl is None:
            domainInControl = self.domainString
        if self.episodes[domainInControl] is None:
            logger.warning("record attempted to be finalized for domain where nothing has been recorded before")
            return

        # print 'Episode Avg_Max_Q', float(self.episode_ave_max_q)/float(self.episodes[domainInControl].size())
        #print 'Episode Avg_Max_Q', np.mean(self.episode_ave_max_q)

        #print 'saving statics'
        #self.saveStats()
        #print self.stats
        #print 'stdVar'
        #print self.stdVar
        #print 'meanVar'
        #print self.meanVar
        #print 'stdMean'
        #print self.stdMean
        #print 'meanMean'
        #print self.meanMean
        # print 'td_error'
        # print self.td_error
        # print 'td_errorVar'
        # print self.td_errorVar

        # normalising total return to -1~1
        reward /= 20.0

        terminal_state, terminal_action = self.convertStateAction(TerminalState(), TerminalAction())

        if self.replay_type == 'vanilla':
            self.episodes[domainInControl].record(state=terminal_state, \
                                                  state_ori=TerminalState(), action=terminal_action, reward=reward,
                                                  terminal=True)
        elif self.replay_type == 'prioritized':
            # heuristically assign 0.0 to Q_s_t_a_t_ and Q_s_tplu1_maxa_, doesn't matter as it is not used
            if True:
                # if self.samplecount >= self.capacity:
                self.episodes[domainInControl].record(state=terminal_state, \
                                                      state_ori=TerminalState(), action=terminal_action, reward=reward, \
                                                      Q_s_t_a_t_=0.0, gamma_Q_s_tplu1_maxa_=0.0, uniform=False,
                                                      terminal=True)
            else:
                self.episodes[domainInControl].record(state=terminal_state, \
                                                      state_ori=TerminalState(), action=terminal_action, reward=reward, \
                                                      Q_s_t_a_t_=0.0, gamma_Q_s_tplu1_maxa_=0.0, uniform=True,
                                                      terminal=True)
        return

    def convertStateAction(self, state, action):
        '''
        nnType = 'dnn'
        #nnType = 'rnn'
        # expand one dimension to match the batch size of 1 at axis 0
        if nnType == 'rnn':
            belief = np.expand_dims(belief,axis=0)
        '''

        if isinstance(state, TerminalState):
            if self.domainUtil.domainString == 'CamRestaurants':
                return [0] * 268, action
            elif self.domainUtil.domainString == 'CamHotels':
                return [0] * 111, action
            elif self.domainUtil.domainString == 'SFRestaurants':
                return [0] * 636, action
            elif self.domainUtil.domainString == 'SFHotels':
                return [0] * 438, action
            elif self.domainUtil.domainString == 'Laptops11':
                return [0] * 257, action
            elif self.domainUtil.domainString == 'TV':
                return [0] * 188, action
        else:
            flat_belief = flatten_belief(state, self.domainUtil)
            self.prev_state_check = flat_belief

            return flat_belief, action

    def nextAction(self, beliefstate):
        '''
        select next action

        :param beliefstate:
        :param hyps:
        :returns: (int) next summary action
        '''
        beliefVec = flatten_belief(beliefstate, self.domainUtil)
        execMask = self.summaryaction.getExecutableMask(beliefstate, self.lastSystemAction)

        if self.exploration_type == 'e-greedy':
            # epsilon greedy
            if self.is_training and utils.Settings.random.rand() < self.epsilon:
                admissible = [i for i, x in enumerate(execMask) if x == 0.0]
                random.shuffle(admissible)
                nextaIdex = admissible[0]
            else:
                action_Q = self.bbqn.predict(np.reshape(beliefVec, (1, len(beliefVec))))  # + (1. / (1. + i + j))
                admissible = np.add(action_Q, np.array(execMask))
                logger.info('action Q...')
                print admissible
                nextaIdex = np.argmax(admissible)

                # add current max Q to self.episode_ave_max_q
                print 'current maxQ', np.max(admissible)
                self.episode_ave_max_q.append(np.max(admissible))

        elif self.exploration_type == 'Boltzman':
            # softmax
            if not self.is_training:
                self.epsilon = 0.001
            # self.epsilon here is served as temperature
            action_Q = self.bbqn.predict(np.reshape(beliefVec, (1, len(beliefVec))))  # + (1. / (1. + i + j))
            action_Q_admissible = np.add(action_Q, np.array(execMask))  # enforce Q of inadmissible actions to be -inf

            action_prob = drlutils.softmax(action_Q_admissible / self.epsilon)
            logger.info('action Q...')
            print action_Q_admissible
            logger.info('action prob...')
            print action_prob
            sampled_prob = np.random.choice(action_prob[0], p=action_prob[0])
            nextaIdex = np.argmax(action_prob[0] == sampled_prob)

        self.stats[nextaIdex] += 1
        summaryAct = self.summaryaction.action_names[nextaIdex]
        beliefstate = beliefstate.getDomainState(self.domainUtil.domainString)
        masterAct = self.summaryaction.Convert(beliefstate, summaryAct, self.lastSystemAction)
        return masterAct, nextaIdex

    def train(self):
        '''
        call this function when the episode ends
        '''

        if not self.is_training:
            logger.info("Not in training mode")
            return
        else:
            logger.info("Update bbqn policy parameters.")

        self.episodecount += 1
        logger.info("Sample Num so far: %s" % (self.samplecount))
        logger.info("Episode Num so far: %s" % (self.episodecount))

        if self.samplecount >= self.minibatch_size * 3 and self.episodecount % self.training_frequency == 0:
            logger.info('start training...')

            s_batch, s_ori_batch, a_batch, r_batch, s2_batch, s2_ori_batch, t_batch, idx_batch, _ = \
                self.episodes[self.domainString].sample_batch()

            # here?
            s_batch = np.vstack([np.expand_dims(x, 0) for x in s_batch])
            s2_batch = np.vstack([np.expand_dims(x, 0) for x in s2_batch])
            # target_q = self.bbqn.predict_target_with_action_maxQ(s2_batch)
            action_q = self.bbqn.predict(s2_batch)
            target_q = self.bbqn.predict_target(s2_batch)

            y_i = []
            for k in xrange(min(self.minibatch_size, self.episodes[self.domainString].size())):
                Q_bootstrap_label = 0
                if t_batch[k]:
                    Q_bootstrap_label = r_batch[k]
                else:
                    if self.q_update == 'single':
                        execMask = self.summaryaction.getExecutableMask(s2_ori_batch[k], a_batch[k])
                        action_Q = target_q[k]
                        admissible = np.add(action_Q, np.array(execMask))
                        # logger.info('action Q...')
                        # print admissible
                        Q_bootstrap_label = r_batch[k] + self.gamma * np.max(admissible)
                    elif self.q_update == 'double':
                        execMask = self.summaryaction.getExecutableMask(s2_ori_batch[k], a_batch[k])
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
            a_batch_one_hot = np.eye(self.action_dim, self.action_dim)[a_batch]

            # Update the critic given the targets
            reshaped_yi = np.vstack([np.expand_dims(x, 0) for x in y_i])
            logger.info("s_batch")
            logger.info(s_batch)
            # reshaped_yi = np.reshape(y_i, (min(self.minibatch_size, self.episodes[self.domainString].size()), 1))
	    #self.bbqn.update_target_network()
            predicted_q_value, _, currentLoss, logLikelihood, varFC2, meanFC2, td_error, KL_div = self.bbqn.train(s_batch, a_batch_one_hot, reshaped_yi, self.episodecount)
#, ct506d,ct5066d, ct507d, ct508d, ct509d, ct510d, ct511d
            # print 'y_i'
            # print y_i
            # print 'currentLoss', currentLoss
            # print 'predict Q'
            # print predicted_q_value
            # print 'loglikelihood', logLikelihood
            # #print 'ct506', np.shape(np.array(td_error))
            # print 'KL Div', KL_div
            #print 'ct506d', np.shape(np.array(ct506d)), ct506d
            #print 'ct5066d', np.shape(np.array(ct5066d)), ct5066d
            #print 'ct507d', np.shape(np.array(ct507d)), ct507d
            #print 'ct508d', np.shape(np.array(ct508d)), ct508d
            #print 'ct509d', np.shape(np.array(ct509d)), ct509d
            #print 'ct510d', np.shape(np.array(ct510d)), ct510d
            #print 'ct511d', np.shape(np.array(ct511d)), ct511d
            #print 'pi_i', pi_i
            #print 'shape ct5066' , logLikelihood.shape()
            self.stdVar += [np.var(np.log(1.0+np.exp(varFC2)))]
            self.meanVar += [np.mean(np.log(1.0+np.exp(varFC2)))]
            self.stdMean += [np.var(meanFC2)]
            self.meanMean += [np.mean(meanFC2)]
            self.td_error += [np.mean(td_error)]
            self.td_errorVar += [np.var(td_error)]

            if self.episodecount % 1 == 0:
                # if self.episodecount % 50 == 0:
                # Update target networks
                self.bbqn.update_target_network()

        self.savePolicyInc()  # self.out_policy_file)

    def savePolicy(self, FORCE_SAVE=False):
        """
        Does not use this, cause it will be called from agent after every episode.
        we want to save the policy only periodically.
        """
        pass

    def savePolicyInc(self, FORCE_SAVE=False):
        """
        save model and replay buffer
        """
        if self.episodecount % self.save_step == 0:
            # save_path = self.saver.save(self.sess, self.out_policy_file+'.ckpt')
            self.bbqn.save_network(self.out_policy_file + '.dqn.ckpt')

            f = open(self.out_policy_file + '.episode', 'wb')
            for obj in [self.samplecount, self.episodes[self.domainString]]:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
            # logger.info("Saving model to %s and replay buffer..." % save_path)

    def loadPolicy(self, filename):
        """
        load model and replay buffer
        """
        # load models
        self.bbqn.load_network(filename + '.dqn.ckpt')

        # load replay buffer
        try:
            print 'laod from: ', filename
            f = open(filename + '.episode', 'rb')
            loaded_objects = []
            for i in range(2):  # load nn params and collected data
                loaded_objects.append(pickle.load(f))
            self.samplecount = int(loaded_objects[0])
            self.episodes[self.domainString] = copy.deepcopy(loaded_objects[1])
            logger.info("Loading both model from %s and replay buffer..." % filename)
            f.close()
        except:
            logger.info("Loading only models...")

    def restart(self):
        self.summaryAct = None
        self.lastSystemAction = None
        self.prevbelief = None
        self.actToBeRecorded = None
        self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * float(
            self.episodeNum + self.episodecount) / float(self.maxiter)
        print 'current eps', self.epsilon
        # self.episodes = dict.fromkeys(OntologyUtils.available_domains, None)
        # self.episodes[self.domainString] = ReplayBuffer(self.capacity, self.randomseed)
        self.episode_ave_max_q = []

# END OF FILE
