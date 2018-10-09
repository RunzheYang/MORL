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
A2CPolicy.py - Advantage Actor-Critic policy
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
import scipy
import scipy.signal
import cPickle as pickle
import random
import utils
from utils.Settings import config as cfg
from utils import ContextLogger, DiaAct

import ontology.FlatOntologyManager as FlatOnt

import tensorflow as tf
from DRL.replay_buffer_episode_a2c import ReplayBufferEpisode
from DRL.replay_prioritised_episode import ReplayPrioritisedEpisode
import DRL.utils as drlutils
import DRL.a2c as a2c
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
            return [0] * 636
        elif domainUtil.domainString == 'SFHotels':
            return [0] * 438
        elif domainUtil.domainString == 'Laptops11':
            return [0] * 257
        elif domainUtil.domainString == 'TV':
            return [0] * 188

    policyfeatures = ['full', 'method', 'discourseAct', 'requested',
                      'lastActionInformNone', 'offerHappened', 'inform_info']

    flat_belief = []
    for feat in policyfeatures:
        add_feature = []
        if feat == 'full':
            # for slot in self.sorted_slots:
            for slot in domainUtil.ontology['informable']:
                for value in domainUtil.ontology['informable'][slot]:  # + ['**NONE**']:
                    add_feature.append(belief['beliefs'][slot][value])

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


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class A2CPolicy(Policy.Policy):
    '''Derived from :class:`Policy`
    '''
    def __init__(self, in_policy_file, out_policy_file, domainString='CamRestaurants', is_training=False):
        super(A2CPolicy, self).__init__(domainString, is_training)

        tf.reset_default_graph()

        self.in_policy_file = in_policy_file
        self.out_policy_file = out_policy_file
        self.is_training = is_training
        self.accum_belief = []
        self.prev_state_check = None

        self.domainString = domainString
        self.domainUtil = FlatOnt.FlatDomainOntology(self.domainString)

        # parameter settings
        self.n_in = 260
        if 0:  # cfg.has_option('dqnpolicy', 'n_in'): #ic304: this was giving me a weird error, disabled it until i can check it deeper
            self.n_in = cfg.getint('dqnpolicy', 'n_in')
        else:
            self.n_in = self.get_n_in(domainString)

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

        self.learning_rate = 0.001
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

        self.capacity = 1000
        if cfg.has_option('dqnpolicy', 'capacity'):
            self.capacity = cfg.getint('dqnpolicy', 'capacity')

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

        self.importance_sampling = False
        if cfg.has_option('dqnpolicy', 'importance_sampling'):
            self.importance_sampling = cfg.getint('dqnpolicy', 'importance_sampling')

        self.training_frequency = 2
        if cfg.has_option('dqnpolicy', 'training_frequency'):
            self.training_frequency = cfg.getint('dqnpolicy', 'training_frequency')

        # domain specific parameter settings (overrides general policy parameter settings)

        if cfg.has_option('dqnpolicy_'+domainString, 'n_in'):
            self.n_in = cfg.getint('dqnpolicy_'+domainString, 'n_in')

        if cfg.has_option('dqnpolicy_'+domainString, 'actor_lr'):
            self.actor_lr = cfg.getfloat('dqnpolicy_'+domainString, 'actor_lr')

        if cfg.has_option('dqnpolicy_'+domainString, 'critic_lr'):
            self.critic_lr = cfg.getfloat('dqnpolicy_'+domainString, 'critic_lr')

        if cfg.has_option('dqnpolicy_'+domainString, 'tau'):
            self.tau = cfg.getfloat('dqnpolicy_'+domainString, 'tau')

        if cfg.has_option('dqnpolicy_'+domainString, 'gamma'):
            self.gamma = cfg.getfloat('dqnpolicy_'+domainString, 'gamma')

        if cfg.has_option('dqnpolicy_'+domainString, 'regularisation'):
            self.regularisation = cfg.get('dqnpolicy_'+domainString, 'regulariser')

        if cfg.has_option('dqnpolicy_'+domainString, 'learning_rate'):
            self.learning_rate = cfg.getfloat('dqnpolicy_'+domainString, 'learning_rate')

        self.exploration_type = 'e-greedy' # Boltzman
        if cfg.has_option('dqnpolicy_'+domainString, 'exploration_type'):
            self.exploration_type = cfg.get('dqnpolicy_'+domainString, 'exploration_type')

        if cfg.has_option('dqnpolicy_'+domainString, 'episodeNum'):
            self.episodeNum = cfg.getfloat('dqnpolicy_'+domainString, 'episodeNum')

        if cfg.has_option('dqnpolicy_'+domainString, 'maxiter'):
            self.maxiter = cfg.getfloat('dqnpolicy_'+domainString, 'maxiter')

        if cfg.has_option('dqnpolicy_'+domainString, 'epsilon'):
            self.epsilon = cfg.getfloat('dqnpolicy_'+domainString, 'epsilon')

        if cfg.has_option('dqnpolicy_'+domainString, 'epsilon_start'):
            self.epsilon_start = cfg.getfloat('dqnpolicy_'+domainString, 'epsilon_start')

        if cfg.has_option('dqnpolicy_'+domainString, 'epsilon_end'):
            self.epsilon_end = cfg.getfloat('dqnpolicy_'+domainString, 'epsilon_end')

        if cfg.has_option('dqnpolicy_'+domainString, 'prior_sample_prob_start'):
            self.priorProbStart = cfg.getfloat('dqnpolicy_'+domainString, 'prior_sample_prob_start')

        if cfg.has_option('dqnpolicy_'+domainString, 'prior_sample_prob_end'):
            self.priorProbEnd = cfg.getfloat('dqnpolicy_'+domainString, 'prior_sample_prob_end')

        if cfg.has_option('dqnpolicy_'+domainString, 'features'):
            logger.info('Features: ' + str(cfg.get('dqnpolicy_'+domainString, 'features')))
            self.policyfeatures = json.loads(cfg.get('dqnpolicy_'+domainString, 'features'))

        if cfg.has_option('dqnpolicy_'+domainString, 'max_k'):
            self.max_k = cfg.getint('dqnpolicy_'+domainString, 'max_k')

        if cfg.has_option('dqnpolicy_'+domainString, 'learning_algorithm'):
            self.learning_algorithm = cfg.get('dqnpolicy_'+domainString, 'learning_algorithm')
            logger.info('Learning algorithm: ' + self.learning_algorithm)

        if cfg.has_option('dqnpolicy_'+domainString, 'minibatch_size'):
            self.minibatch_size = cfg.getint('dqnpolicy_'+domainString, 'minibatch_size')

        if cfg.has_option('dqnpolicy_'+domainString, 'capacity'):
            self.capacity = cfg.getint('dqnpolicy_'+domainString,'capacity')

        if cfg.has_option('dqnpolicy_'+domainString, 'replay_type'):
            self.replay_type = cfg.get('dqnpolicy_'+domainString, 'replay_type')

        if cfg.has_option('dqnpolicy_'+domainString, 'architecture'):
            self.architecture = cfg.get('dqnpolicy_'+domainString, 'architecture')

        if cfg.has_option('dqnpolicy_'+domainString, 'q_update'):
            self.q_update = cfg.get('dqnpolicy_'+domainString, 'q_update')

        if cfg.has_option('dqnpolicy_'+domainString, 'h1_size'):
            self.h1_size = cfg.getint('dqnpolicy_'+domainString, 'h1_size')

        if cfg.has_option('dqnpolicy_'+domainString, 'h2_size'):
            self.h2_size = cfg.getint('dqnpolicy_'+domainString, 'h2_size')

        if cfg.has_option('policy_' + domainString, 'save_step'):
            self.save_step = cfg.getint('policy_' + domainString, 'save_step')

        if cfg.has_option('dqnpolicy_'+domainString, 'importance_sampling'):
            self.importance_sampling = cfg.getint('dqnpolicy_'+domainString, 'importance_sampling')

        if cfg.has_option('dqnpolicy_'+domainString, 'training_frequency'):
            self.training_frequency = cfg.getint('dqnpolicy_'+domainString, 'training_frequency')

        self.episode_ave_max_q = []
        self.mu_prob = 0.  # behavioral policy

        os.environ["CUDA_VISIBLE_DEVICES"]=""

        # init session
        self.sess = tf.Session()
        with tf.device("/cpu:0"):
            np.random.seed(self.randomseed)
            tf.set_random_seed(self.randomseed)

            # initialise an replay buffer
            if self.replay_type == 'vanilla':
                self.episodes[self.domainString] = ReplayBufferEpisode(self.capacity, self.minibatch_size, self.randomseed)
            elif self.replay_type == 'prioritized':
                self.episodes[self.domainString] = ReplayPrioritisedEpisode(self.capacity, self.minibatch_size, self.randomseed)

            self.samplecount = 0
            self.episodecount = 0

            # construct the models
            self.state_dim = self.n_in
            self.summaryaction = SummaryAction.SummaryAction(domainString)
            self.action_dim = len(self.summaryaction.action_names)
            action_bound = len(self.summaryaction.action_names)
            self.stats = [0 for _ in range(self.action_dim)]

            self.a2c = a2c.A2CNetwork(self.sess, self.state_dim, self.action_dim, \
                self.critic_lr, self.tau, action_bound, self.architecture, self.h1_size, self.h2_size, self.is_training)
            
            # when all models are defined, init all variables
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

            self.loadPolicy(self.in_policy_file)
            print 'loaded replay size: ', self.episodes[self.domainString].size()

    def get_n_in(self, domain_string):
        if domain_string == 'CamRestaurants':
            return 268
        elif domain_string == 'CamHotels':
            return 111
        elif domain_string == 'SFRestaurants':
            return 636
        elif domain_string == 'SFHotels':
            return 438
        elif domain_string == 'Laptops6':
            return 268 # ic340: this is wrong
        elif domain_string == 'Laptops11':
            return 257
        elif domain_string is 'TV':
            return 188
        else:
            print 'DOMAIN {} SIZE NOT SPECIFIED, PLEASE DEFINE n_in'.format(domain_string)

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
            self.actToBeRecorded = self.summaryAct

        if state is None:
            state = self.prevbelief
        if action is None:
            action = self.actToBeRecorded

        cState, cAction = self.convertStateAction(state, action)

        # normalising total return to -1~1
        reward /= 20.0

        value = self.a2c.predict_value([cState])
        policy_mu = self.mu_prob  # self.a2c.getPolicy([cState])[0][0][cAction]

        if self.replay_type == 'vanilla':
            self.episodes[domainInControl].record(state=cState, \
                    state_ori=state, action=cAction, reward=reward, value=value[0][0], distribution=policy_mu)
        elif self.replay_type == 'prioritized':
            self.episodes[domainInControl].record(state=cState, \
                    state_ori=state, action=cAction, reward=reward, value=value[0][0], distribution=policy_mu)

        self.actToBeRecorded = None
        self.samplecount += 1

    def finalizeRecord(self, reward, domainInControl=None):
        if domainInControl is None:
            domainInControl = self.domainString
        if self.episodes[domainInControl] is None:
            logger.warning("record attempted to be finalized for domain where nothing has been recorded before")
            return

        #print 'Episode Avg_Max_Q', float(self.episode_ave_max_q)/float(self.episodes[domainInControl].size())
        #print 'Episode Avg_Max_Q', np.mean(self.episode_ave_max_q)
        #print self.stats

        # normalising total return to -1~1
        reward /= 20.0

        terminal_state, terminal_action = self.convertStateAction(TerminalState(), TerminalAction())

        value = 0.0  # not effect on experience replay

        def calculate_advantage(r_episode, v_episode):
            #########################################################################  
            # Here we take the rewards and values from the rollout, and use them to 
            # generate the advantage and discounted returns. 
            # The advantage function uses "Generalized Advantage Estimation"
            bootstrap_value = 0.0
            self.v_episode_plus = np.asarray(v_episode + [bootstrap_value])
            advantage = r_episode + self.gamma * self.v_episode_plus[1:] - self.v_episode_plus[:-1]
            advantage = discount(advantage,self.gamma)
            #########################################################################  
            return advantage

        if self.replay_type == 'vanilla':
            self.episodes[domainInControl].record(state=terminal_state, \
                    state_ori=TerminalState(), action=terminal_action, reward=reward, value=value, terminal=True, distribution=None)
        elif self.replay_type == 'prioritized':
            episode_r, episode_v = self.episodes[domainInControl].record_final_and_get_episode(state=terminal_state, \
                    state_ori=TerminalState(), action=terminal_action, reward=reward, value=value, terminal=True, distribution=None)
            
            # TD_error is a list of td error in the current episode
            TD_error = calculate_advantage(episode_r, episode_v)
            episodic_TD = np.mean(np.absolute(TD_error))
            print 'episodic_TD'
            print episodic_TD
            print 'total TD', self.episodes[self.domainString].tree.total()
            self.episodes[domainInControl].insertPriority(episodic_TD)

    def convertStateAction(self, state, action):
        if isinstance(state, TerminalState):
            if self.domainUtil.domainString == 'CamRestaurants':
                return [0] * 268, action
            elif self.domainUtil.domainString == 'CamHotels':
                return [0] * 111, action
            elif self.domainUtil.domainString == 'SFRestaurants':
                return [0] * 633, action
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

            action_prob, value = self.a2c.predict_action_value(np.reshape(beliefVec, (1, len(beliefVec))))# + (1. / (1. + i + j))
            admissibleCnt = [i for i, x in enumerate(execMask) if x == 0.0]
            admissible = np.add(action_prob, np.array(execMask))
            greedyNextaIdex = np.argmax(admissible)

            # epsilon greedy
            if self.is_training and utils.Settings.random.rand() < self.epsilon:
                admissible = [i for i, x in enumerate(execMask) if x == 0.0]
                random.shuffle(admissible)
                nextaIdex = admissible[0]

                # Importance sampling
                if nextaIdex == greedyNextaIdex:
                    self.mu_prob = self.epsilon / float(self.action_dim) + 1 - self.epsilon
                else:
                    self.mu_prob = self.epsilon / float(self.action_dim)
            else:
                """
                action_prob, value = self.a2c.predict_action_value(np.reshape(beliefVec, (1, len(beliefVec))))# + (1. / (1. + i + j))
                admissible = np.add(action_prob, np.array(execMask))
                logger.info('action Q...')
                print admissible
                print value
                nextaIdex = np.argmax(admissible)
                """
                nextaIdex = greedyNextaIdex

                # add current max Q to self.episode_ave_max_q
                #print 'current maxQ', np.max(admissible)
                self.episode_ave_max_q.append(np.max(admissible))

                # Importance sampling
                self.mu_prob = self.epsilon / float(self.action_dim) + 1 - self.epsilon

        elif self.exploration_type == 'Boltzman':
            # softmax
            if not self.is_training:
                self.epsilon = 0.001
            # self.epsilon here is served as temperature
            action_prob, value = self.a2c.predict_action_value(np.reshape(beliefVec, (1, len(beliefVec))))# + (1. / (1. + i + j))
            action_Q_admissible = np.add(action_prob, np.array(execMask)) # enforce Q of inadmissible actions to be -inf

            action_prob = drlutils.softmax(action_Q_admissible/self.epsilon)
            #logger.info('action Q...')
            #print action_Q_admissible
            #logger.info('action prob...')
            #print action_prob
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
            logger.info("Update a2c policy parameters.")

        self.episodecount += 1
        logger.info("Sample Num so far: %s" % self.samplecount)
        logger.info("Episode Num so far: %s" % self.episodecount)

        if self.samplecount >= self.minibatch_size * 3 and self.episodecount % self.training_frequency == 0:
            logger.info('start training...')

            s_batch, s_ori_batch, a_batch, r_batch, s2_batch, s2_ori_batch, t_batch, idx_batch, v_batch, mu_policy = \
                self.episodes[self.domainString].sample_batch()

            discounted_r_batch = []
            advantage_batch = []

            def weightsImportanceSampling(mu_policy, r_batch):
                mu_policy = np.asarray(mu_policy)
                mu_cum = []
                lenghts = []  # to properly divde on dialogues pi_policy later on
                for mu in mu_policy:
                    lenghts.append(len(mu))
                    mu = np.asarray(mu).astype(np.longdouble)
                    mu_cum.append(np.cumprod(mu[::-1])[::-1])  # going forward with cumulative product
                # mu_cum = np.concatenate(np.array(mu_cum), axis=0).tolist()
                mu_policy = np.concatenate(np.array(mu_policy), axis=0).tolist()  # concatenate all behavioral probs
                lengths = np.cumsum(lenghts)  # time steps for ends of dialogues

                pi_policy = self.a2c.getPolicy(np.concatenate(np.array(s_batch), axis=0).tolist())[0]  # policy given s_t
                columns = np.asarray([np.concatenate(a_batch, axis=0).tolist()]).astype(int)  # actions taken at s_t
                rows = np.asarray([ii for ii in range(len(pi_policy))])
                pi_policy = pi_policy[rows, columns][0].astype(np.longdouble)

                #####################################
                # Weights for importance sampling
                # it goes through each dialogue and computes in reverse order cumulative prod:
                # rho_n = pi_n / mu_n
                # ...
                # rho_1 = pi_1 / mu_1 *  ... * pi_n / mu_n
                # using dialogue and weight_cum lists
                #####################################

                rho_forward = []  # rho_forward from eq. 3.3 (the first one)
                rho_whole = []  # product across the whole dialogue from eq. 3.3 (the second one)
                #pi_cum2 = []  # stats to compare
                #mu_cum2 = []  # stats to compare
                #pi_cum = []  # stats to compare

                # Precup version
                r_vector = np.concatenate(np.array(r_batch), axis=0).tolist()
                r_weighted = []

                for ii in range(len(lengths) - 1):  # over dialogues
                    weight_cum = 1.
                    dialogue = []

                    # first case
                    if ii == 0:
                        for pi, mu in zip(pi_policy[0:lengths[0]], mu_policy[0:lengths[0]]):
                            weight_cum *= pi / mu
                            dialogue.append(weight_cum)

                        dialogue = np.array(dialogue)

                        if self.importance_sampling:
                            dialogue = np.clip(dialogue, -1, 1)
                        else:
                            dialogue = np.ones(dialogue.shape)
                        dialogue = dialogue.tolist()

                        rho_forward.extend(dialogue)
                        rho_whole.extend(np.ones(len(dialogue)) * dialogue[-1])
                        r_weighted.extend(r_vector[0:lengths[0]] * np.asarray(dialogue))
                        dialogue = []

                    for pi, mu in zip(pi_policy[lengths[ii]:lengths[ii + 1]], mu_policy[lengths[ii]:lengths[ii + 1]]):
                        weight_cum *= pi / mu
                        dialogue.append(weight_cum)

                    dialogue = np.array(dialogue)
                    if self.importance_sampling:
                        dialogue = np.clip(dialogue, -1, 1)
                    else:
                        dialogue = np.ones(dialogue.shape)
                    dialogue = dialogue.tolist()

                    rho_forward.extend(dialogue)
                    rho_whole.extend(np.ones(len(dialogue)) * dialogue[-1])
                    r_weighted.extend(r_vector[lengths[ii]: lengths[ii + 1]] * np.asarray(dialogue))

                # go back to original form:
                ind = 0
                r_new = copy.deepcopy(r_batch)
                for id, batch in enumerate(r_new):
                    for id2, _ in enumerate(batch):
                        r_new[id][id2] = r_weighted[ind]
                        ind += 1

                # ONE STEP WEIGHTS
                weights = np.asarray(pi_policy) / np.asarray(mu_policy)
                if self.importance_sampling:
                    weights = np.clip(weights, -1, 1)
                else:
                    weights = np.ones(weights.shape)

                return weights, rho_forward, rho_whole, r_new

            weights, rho_forward, rho_whole, r_new = weightsImportanceSampling(mu_policy, r_batch)
            weights = np.nan_to_num(weights)
            rho_forward = np.nan_to_num(rho_forward)
            #rho_whole = np.nan_to_num(rho_whole)
 
            """
            weights[weights > 1] = 1
            rho_forward[rho_forward > 1] = 1
            rho_whole[rho_whole >1] = 1

            print weights.size
            print weights
            print max(weights)
            print min(weights)
            print rho_forward.size
            print rho_forward
            print max(rho_forward)
            print min(rho_forward)
            print rho_whole.size
            print rho_whole
            print max(rho_whole)
            print min(rho_whole)
            print len(r_new)
            print r_new
            print max(r_new)
            print min(r_new)
            """
            # Precup version
            # rho_whole = r_new

            def calculate_advantage(r_episode, v_episode):
                #########################################################################
                # Here we take the rewards and values from the rolloutv, and use them to
                # generate the advantage and discounted returns.
                # The advantage function uses "Generalized Advantage Estimation"
                bootstrap_value = 0.0
                self.v_episode_plus = np.asarray(v_episode + [bootstrap_value])
                # change sth here
                advantage = r_episode + self.gamma * self.v_episode_plus[1:] - self.v_episode_plus[:-1]
                advantage = discount(advantage, self.gamma)
                #########################################################################
                return advantage

            def calculate_discountR(r_episode, idx):
                #########################################################################
                # Here we take the rewards and values from the rolloutv, and use them to
                # generate the advantage and discounted returns.
                # The advantage function uses "Generalized Advantage Estimation"
                bootstrap_value = 0.0
                # r_episode rescale by rhos?
                self.r_episode_plus = np.asarray(r_episode[idx:] + [bootstrap_value])
                if self.importance_sampling:
                    self.r_episode_plus = self.r_episode_plus
                else:
                    self.r_episode_plus = self.r_episode_plus / rho_forward[idx]
                discounted_r_episode = discount(self.r_episode_plus, self.gamma)[:-1]
                #########################################################################
                return discounted_r_episode[0]
            
            if self.replay_type == 'prioritized':
                for item_r, item_v, item_idx in zip(r_new, v_batch, idx_batch):
                    rlist = []
                    for idx in range(len(item_r)):
                        r = calculate_discountR(item_r, idx)
                        rlist.append(r)

                    a = calculate_advantage(item_r, item_v)

                    # flatten nested numpy array and turn it into list
                    discounted_r_batch += rlist
                    advantage_batch += a.tolist()

                    # update the sum-tree
                    # update the TD error of the samples (episode) in the minibatch
                    episodic_TD_error = np.mean(np.absolute(a))
                    self.episodes[self.domainString].update(item_idx, episodic_TD_error)
            else:
                for item_r, item_v in zip(r_new, v_batch):
                    rlist = []
                    for idx in range(len(item_r)):
                        r = calculate_discountR(item_r, idx)
                        rlist.append(r)

                    a = calculate_advantage(item_r, item_v)

                    # flatten nested numpy array and turn it into list
                    discounted_r_batch += rlist
                    advantage_batch += a.tolist()

            batch_size = len(s_batch)

            """
            print 'w', weights
            print 'r', r_new
            print 'discount_r', discounted_r_batch
            print 'advantage', advantage_batch

            print 'max_discount_r', max(discounted_r_batch)
            print 'max_discount_ad', max(advantage_batch)

            print s_batch
            yo = np.concatenate(np.array(s_batch), axis=0).tolist()
            print yo
            print a_batch
            print np.concatenate(a_batch).ravel().tolist()

            #s_batch = np.vstack([np.expand_dims(x, 0) for x in s_batch])
            #discounted_r_batch = np.vstack([np.expand_dims(x, 0) for x in discounted_r_batch])
            #advantage_batch = np.vstack([np.expand_dims(x, 0) for x in advantage_batch])
            print 'Reward.....'
            print discounted_r_batch
            print 'Advantage.....'
            print advantage_batch
            """
            # change index-based a_batch to one-hot-based a_batch
            a_batch_one_hot = np.eye(self.action_dim)[np.concatenate(a_batch, axis=0).tolist()]

            #policy_test  = self.a2c.getPolicy(np.concatenate(np.array(s_batch), axis=0).tolist(), a_batch_one_hot, discounted_r_batch, advantage_batch)
            #print 'policy looks like', policy_test

            """
            print 'rho_forward', rho_forward
            print 'rho_whole', rho_whole
            print 'dis_r', discounted_r_batch
            print 'adv', advantage_batch
            diff = rho_forward * np.square(discounted_r_batch-np.reshape(advantage_batch, [-1]))
            print 'diff', diff
            print 'max', max(diff)
            print 'min', min(diff)
            print 'loss sum', 0.5 * np.sum(diff)
            print 'loss mean', 0.5 * np.mean(diff)
            """

            if self.importance_sampling:
                discounted_r_batch = np.clip(discounted_r_batch, -2, 2)

            value_loss, policy_loss, entropy, all_loss = \
                    self.a2c.getLoss(np.concatenate(np.array(s_batch), axis=0).tolist(), a_batch_one_hot, discounted_r_batch, advantage_batch, weights, rho_forward)#, rho_whole)

            norm_v_l, norm_p_l, ent, norm_all_l = \
                    value_loss/float(batch_size), policy_loss/float(batch_size), \
                    entropy/float(batch_size), all_loss/float(batch_size)
            
            #print 'normalised from %d episodes' %(batch_size)
            #print 'value loss', norm_v_l
            #print 'policy loss', norm_p_l
            #print 'entropy', ent
            #print 'total loss', norm_all_l

            value_loss, policy_loss, entropy, all_loss, optimise = \
                    self.a2c.train(np.concatenate(np.array(s_batch), axis=0).tolist(), a_batch_one_hot,
                                   discounted_r_batch, advantage_batch, weights, rho_forward)

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
            self.a2c.save_network(self.out_policy_file+'.a2c.ckpt')

            f = open(self.out_policy_file+'.episode', 'wb')
            for obj in [self.samplecount, self.episodes[self.domainString]]:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
            #logger.info("Saving model to %s and replay buffer..." % save_path)

    def loadPolicy(self, filename):
        """
        load model and replay buffer
        """
        # load models
        self.a2c.load_network(filename+'.a2c.ckpt')

        # load replay buffer
        try:
            print 'load from: ', filename
            f = open(filename+'.episode', 'rb')
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
        self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * float(self.episodeNum+self.episodecount) / float(self.maxiter)
        #print 'current eps', self.epsilon
        self.episode_ave_max_q = []

#END OF FILE
