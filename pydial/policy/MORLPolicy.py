###############################################################################
# PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################

'''
MORL-Policy.py - multi-objective reinforcement learning policy
'''

import copy
import os
import sys
import json
import random
import utils
from collections import namedtuple
from collections import deque
from utils.Settings import config as cfg
from utils import ContextLogger, DiaAct

import random
import torch
import copy
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import ontology.FlatOntologyManager as FlatOnt
import DRL.utils as drlutils
import DMORL.naive as naive
import DMORL.envelope as envelope
import Policy
import SummaryAction
from Policy import TerminalAction, TerminalState
from policy.feudalRL.DIP_parametrisation import DIP_state
from utils.monitor import Monitor

logger = utils.ContextLogger.getLogger('')

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

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


class MORLPolicy(Policy.Policy):
    '''Derived from :class:`Policy`
    '''

    def __init__(self, in_policy_file, out_policy_file, domainString='CamRestaurants', is_training=False,
                 action_names=None):
        super(MORLPolicy, self).__init__(domainString, is_training)

        self.domainString = domainString
        self.domainUtil = FlatOnt.FlatDomainOntology(self.domainString)
        self.in_policy_file = in_policy_file
        self.out_policy_file = out_policy_file
        self.is_training = is_training
        self.accum_belief = []

        self.domainUtil = FlatOnt.FlatDomainOntology(self.domainString)
        self.prev_state_check = None

        # parameter settings
        if 0:  # cfg.has_option('morlpolicy', 'n_in'): #ic304: this was giving me a weird error, disabled it until i can check it deeper
            self.n_in = cfg.getint('morlpolicy', 'n_in')
        else:
            self.n_in = self.get_n_in(domainString)

        self.n_rew = 1
        if cfg.has_option('morlpolicy', 'n_rew'):
            self.n_rew = cfg.getint('morlpolicy', 'n_rew')

        self.lr = 0.001
        if cfg.has_option('morlpolicy', 'learning_rate'):
            self.lr = cfg.getfloat('morlpolicy', 'learning_rate')

        self.epsilon = 0.5
        if cfg.has_option('morlpolicy', 'epsilon'):
            self.epsilon = cfg.getfloat('morlpolicy', 'epsilon')

        self.epsilon_decay = True
        if cfg.has_option('morlpolicy', 'epsilon_decay'):
            self.epsilon_decay = cfg.getboolean('morlpolicy', 'epsilon_decay')

        self.randomseed = 1234
        if cfg.has_option('GENERAL', 'seed'):
            self.randomseed = cfg.getint('GENERAL', 'seed')

        self.gamma = 1.0
        if cfg.has_option('morlpolicy', 'gamma'):
            self.gamma = cfg.getfloat('morlpolicy', 'gamma')

        self.weight_num = 32
        if cfg.has_option('morlpolicy', 'weight_num'):
            self.weight_num = cfg.getint('morlpolicy', 'weight_num')

        self.episode_num = 1000
        if cfg.has_option('morlpolicy', 'episode_num'):
            self.episode_num = cfg.getfloat('morlpolicy', 'episode_num')

        self.optimizer = "Adam"
        if cfg.has_option('morlpolicy', 'optimizer'):
            self.optimizer = cfg.get('morlpolicy', 'optimizer')

        self.save_step = 100
        if cfg.has_option('policy', 'save_step'):
            self.save_step = cfg.getint('policy', 'save_step')

        self.update_freq = 50
        if cfg.has_option('morlpolicy', 'update_freq'):
            self.update_freq = cfg.getint('morlpolicy', 'update_freq')

        self.policyfeatures = []
        if cfg.has_option('morlpolicy', 'features'):
            logger.info('Features: ' + str(cfg.get('morlpolicy', 'features')))
            self.policyfeatures = json.loads(cfg.get('morlpolicy', 'features'))

        self.algorithm = 'naive'
        if cfg.has_option('morlpolicy', 'algorithm'):
            self.algorithm = cfg.get('morlpolicy', 'algorithm')
            logger.info('Learning algorithm: ' + self.algorithm)

        self.batch_size = 32
        if cfg.has_option('morlpolicy', 'batch_size'):
            self.batch_size = cfg.getint('morlpolicy', 'batch_size')

        self.mem_size = 1000
        if cfg.has_option('morlpolicy', 'mem_size'):
            self.mem_size = cfg.getint('morlpolicy', 'mem_size')

        self.training_freq = 1
        if cfg.has_option('morlpolicy', 'training_freq'):
            self.training_freq = cfg.getint('morlpolicy', 'training_freq')

        # set beta for envelope algorithm
        self.beta = 0.1
        if cfg.has_option('morlpolicy', 'beta'):
            self.beta = cfg.getfloat('morlpolicy', 'beta')
        self.beta_init = self.beta
        self.beta_uplim = 1.00
        self.tau = 1000.
        self.beta_expbase = float(np.power(self.tau * (self.beta_uplim - self.beta), 1. / (self.episode_num+1)))
        self.beta_delta = self.beta_expbase / self.tau
        self.beta -= self.beta_delta

        # using homotopy method for optimization
        self.homotopy = False
        if cfg.has_option('morlpolicy', 'homotopy'):
            self.homotopy = cfg.getboolean('morlpolicy', 'homotopy')

        self.epsilon_delta = (self.epsilon - 0.05) / self.episode_num

        self.episodecount = 0

        # construct the models
        self.state_dim = self.n_in
        self.summaryaction = SummaryAction.SummaryAction(domainString)
        if action_names is None:
            self.action_names = self.summaryaction.action_names
        else:
            self.action_names = action_names
        self.action_dim = len(self.action_names)
        self.stats = [0 for _ in range(self.action_dim)]
        self.reward_dim = self.n_rew

        model = None
        if self.algorithm == 'naive':
            model = naive.NaiveLinearCQN(self.state_dim, self.action_dim, self.reward_dim)
        elif self.algorithm == 'envelope':
            model = envelope.EnvelopeLinearCQN(self.state_dim, self.action_dim, self.reward_dim)

        self.model_ = model
        self.model = copy.deepcopy(model)

        # initialize memory
        self.trans_mem = deque()
        self.trans = namedtuple('trans', ['s', 'a', 's_', 'r', 'd', 'ms', 'ms_'])
        self.priority_mem = deque()
        self.mem_last_state = None
        self.mem_last_action = None
        self.mem_last_mask = None
        self.mem_cur_state = None
        self.mem_cur_action = None
        self.mem_cur_mask = None

        if self.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model_.parameters(), lr=self.lr)

        try:
            self.loadPolicy(self.in_policy_file)
        except:
            logger.info("No previous model found...")

        self.w_kept = None
        self.update_count = 0
        if self.is_training:
            self.model_.train()
        if use_cuda:
            self.model.cuda()
            self.model_.cuda()

        self.monitor = None

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
            return 268  # ic340: this is wrong
        elif domain_string == 'Laptops11':
            return 257
        elif domain_string is 'TV':
            return 188
        else:
            print 'DOMAIN {} SIZE NOT SPECIFIED, PLEASE DEFINE n_in'.format(domain_string)

    def act_on(self, state, preference=None):
        if self.lastSystemAction is None and self.startwithhello:
            systemAct, nextaIdex = 'hello()', -1
        else:
            systemAct, nextaIdex = self.nextAction(state, preference)
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

        execMask = self.summaryaction.getExecutableMask(state, cAction)
        execMask = torch.Tensor(execMask).type(FloatTensor)

        # # normalising total return to -1~1
        # reward /= 20.0

        self.mem_last_state = self.mem_cur_state
        self.mem_last_action = self.mem_cur_action
        self.mem_last_mask = self.mem_cur_mask
        self.mem_cur_state = np.vstack([np.expand_dims(x, 0) for x in [cState]])
        # self.mem_cur_action = np.eye(self.action_dim, self.action_dim)[[cAction]]
        self.mem_cur_action = cAction
        self.mem_cur_mask = execMask

        state = self.mem_last_state
        action = self.mem_last_action
        next_state = self.mem_cur_state
        terminal = False

        if state is not None and action is not None:
            self.trans_mem.append(self.trans(
                torch.from_numpy(state).type(FloatTensor),  # state
                action,  # action
                torch.from_numpy(next_state).type(FloatTensor),  # next state
                torch.from_numpy(reward).type(FloatTensor),  # reward
                terminal,  # terminal
                self.mem_last_mask,  # action mask
                self.mem_cur_mask))  # next action mask

            # randomly produce a preference for calculating priority
            # preference = self.w_kept
            preference = torch.randn(self.model_.reward_size)
            preference = (torch.abs(preference) / torch.norm(preference, p=1)).type(FloatTensor)

            state = torch.from_numpy(state).type(FloatTensor)

            _, q = self.model_(Variable(state, requires_grad=False),
                               Variable(preference.unsqueeze(0), requires_grad=False),
                               execmask=Variable(self.mem_last_mask.unsqueeze(0), requires_grad=False))

            q = q[0, action].data

            if self.algorithm == 'naive':
                wr = preference.dot(torch.from_numpy(reward).type(FloatTensor))
                if not terminal:
                    next_state = torch.from_numpy(next_state).type(FloatTensor)
                    hq, _ = self.model_(Variable(next_state, requires_grad=False),
                                        Variable(preference.unsqueeze(0), requires_grad=False),
                                        execmask=Variable(self.mem_cur_mask.unsqueeze(0), requires_grad=False))
                    hq = hq.data[0]
                    p = abs(wr + self.gamma * hq - q)
                else:
                    self.w_kept = None
                    # if self.epsilon_decay:
                    #     self.epsilon -= self.epsilon_delta
                    p = abs(wr - q)
            elif self.algorithm == 'envelope':
                wq = preference.dot(q)
                wr = preference.dot(torch.from_numpy(reward).type(FloatTensor))
                if not terminal:
                    next_state = torch.from_numpy(next_state).type(FloatTensor)
                    hq, _ = self.model_(Variable(next_state, requires_grad=False),
                                        Variable(preference.unsqueeze(0), requires_grad=False),
                                        execmask=Variable(self.mem_cur_mask.unsqueeze(0), requires_grad=False))
                    hq = hq.data[0]
                    whq = preference.dot(hq)
                    p = abs(wr + self.gamma * whq - wq)
                else:
                    self.w_kept = None
                    # if self.epsilon_decay:
                    #     self.epsilon -= self.epsilon_delta
                    # if self.homotopy:
                    #     self.beta += self.beta_delta
                    #     self.beta_delta = (self.beta - self.beta_init) * self.beta_expbase + self.beta_init - self.beta
                    p = abs(wr - wq)
            p += 1e-5

            self.priority_mem.append(
                p
            )
            if len(self.trans_mem) > self.mem_size:
                self.trans_mem.popleft()
                self.priority_mem.popleft()

        self.actToBeRecorded = None

    def finalizeRecord(self, reward, domainInControl=None):
        if domainInControl is None:
            domainInControl = self.domainString
        if self.episodes[domainInControl] is None:
            logger.warning("record attempted to be finalized for domain where nothing has been recorded before")
            return

        # # normalising total return to -1~1
        # reward /= 20.0

        terminal_state, terminal_action = self.convertStateAction(TerminalState(), TerminalAction())

        # # normalising total return to -1~1
        # reward /= 20.0

        self.mem_last_state = self.mem_cur_state
        self.mem_last_action = self.mem_cur_action
        self.mem_last_mask = self.mem_cur_mask
        self.mem_cur_state = np.vstack([np.expand_dims(x, 0) for x in [terminal_state]])
        self.mem_cur_action = None
        self.mem_cur_mask = torch.zeros(self.action_dim).type(FloatTensor)

        state = self.mem_last_state
        action = self.mem_last_action
        next_state = self.mem_cur_state
        terminal = True

        if state is not None:
            self.trans_mem.append(self.trans(
                torch.from_numpy(state).type(FloatTensor),  # state
                action,  # action
                torch.from_numpy(next_state).type(FloatTensor),  # next state
                torch.from_numpy(reward).type(FloatTensor),  # reward
                terminal, # terminal
                self.mem_last_mask,  # action mask
                self.mem_cur_mask))  # next action mask

            # randomly produce a preference for calculating priority
            # preference = self.w_kept
            preference = torch.randn(self.model_.reward_size)
            preference = (torch.abs(preference) / torch.norm(preference, p=1)).type(FloatTensor)

            state = torch.from_numpy(state).type(FloatTensor)

            _, q = self.model_(Variable(state, requires_grad=False),
                               Variable(preference.unsqueeze(0), requires_grad=False))

            q = q.data[0, action]

            if self.algorithm == 'naive':
                wr = preference.dot(torch.from_numpy(reward).type(FloatTensor))
                if not terminal:
                    next_state = torch.from_numpy(next_state).type(FloatTensor)
                    hq, _ = self.model_(Variable(next_state, requires_grad=False),
                                        Variable(preference.unsqueeze(0), requires_grad=False))
                    hq = hq.data[0]
                    p = abs(wr + self.gamma * hq - q)
                else:
                    self.w_kept = None
                    # if self.epsilon_decay:
                    #     self.epsilon -= self.epsilon_delta
                    p = abs(wr - q)
            elif self.algorithm == 'envelope':
                wq = preference.dot(q)
                wr = preference.dot(torch.from_numpy(reward).type(FloatTensor))
                if not terminal:
                    next_state = torch.from_numpy(next_state).type(FloatTensor)
                    hq, _ = self.model_(Variable(next_state, requires_grad=False),
                                        Variable(preference.unsqueeze(0), requires_grad=False))
                    hq = hq.data[0]
                    whq = preference.dot(hq)
                    p = abs(wr + self.gamma * whq - wq)
                else:
                    self.w_kept = None
                    # if self.epsilon_decay:
                    #     self.epsilon -= self.epsilon_delta
                    # if self.homotopy:
                    #     self.beta += self.beta_delta
                    #     self.beta_delta = (self.beta - self.beta_init) * self.beta_expbase + self.beta_init - self.beta
                    p = abs(wr - wq)

            p += 1e-5

            self.priority_mem.append(
                p
            )
            if len(self.trans_mem) > self.mem_size:
                self.trans_mem.popleft()
                self.priority_mem.popleft()

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

    def convertDIPStateAction(self, state, action):
        '''

        '''
        if isinstance(state, TerminalState):
            return [0] * 89, action

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

    def nextAction(self, beliefstate, preference=None):
        '''
        select next action

        :param beliefstate:
        :param preference:
        :returns: (int) next summary action
        '''
        beliefVec = flatten_belief(beliefstate, self.domainUtil)
        execMask = self.summaryaction.getExecutableMask(beliefstate, self.lastSystemAction)
        execMask = torch.Tensor(execMask).type(FloatTensor)

        if preference is None:
            if self.w_kept is None:
                self.w_kept = torch.randn(self.model_.reward_size)
                self.w_kept = (torch.abs(self.w_kept) / torch.norm(self.w_kept, p=1)).type(FloatTensor)
            preference = self.w_kept

        if self.is_training and (len(self.trans_mem) < self.batch_size*10 or torch.rand(1)[0] < self.epsilon):
            admissible = [i for i, x in enumerate(execMask) if x == 0.0]
            random.shuffle(admissible)
            nextaIdex = admissible[0]
        else:
            state = np.reshape(beliefVec, (1, len(beliefVec)))
            state = torch.from_numpy(state).type(FloatTensor)
            if self.algorithm == 'naive':
                _, Q = self.model_(
                        Variable(state, requires_grad=False),
                        Variable(preference.unsqueeze(0), requires_grad=False),
                        Variable(execMask.unsqueeze(0), requires_grad=False))
                nextaIdex = np.argmax(Q.detach().cpu().numpy())
            elif self.algorithm == 'envelope':
                _, Q = self.model_(
                    Variable(state, requires_grad=False),
                    Variable(preference.unsqueeze(0), requires_grad=False),
                    execmask=Variable(execMask.unsqueeze(0), requires_grad=False))
                Q = Q.view(-1, self.model_.reward_size)
                Q = torch.mv(Q.data, preference)
                action = Q.max(0)[1].cpu().numpy()
                nextaIdex = int(action)

        self.stats[nextaIdex] += 1
        summaryAct = self.action_names[nextaIdex]
        beliefstate = beliefstate.getDomainState(self.domainUtil.domainString)
        masterAct = self.summaryaction.Convert(beliefstate, summaryAct, self.lastSystemAction)

        return masterAct, nextaIdex

    def sample(self, pop, pri, k):
        pri = np.array(pri).astype(np.float)
        inds = np.random.choice(
            range(len(pop)), k,
            replace=False,
            p=pri / pri.sum()
        )
        return [pop[i] for i in inds]

    def actmsk(self, num_dim, index):
        mask = ByteTensor(num_dim).zero_()
        mask[index] = 1
        return mask.unsqueeze(0)

    def nontmlinds(self, terminal_batch):
        mask = ByteTensor(terminal_batch)
        inds = torch.arange(0, len(terminal_batch)).type(LongTensor)
        inds = inds[mask.eq(0)]
        return inds

    def train(self):
        '''
        call this function when the episode ends
        '''
        self.episodecount +=1
        if self.monitor is None:
            self.monitor = Monitor("-" + self.algorithm)

        if not self.is_training:
            logger.info("Not in training mode")
            return
        else:
            logger.info("Update naive morl policy parameters.")

        logger.info("Episode Num so far: %s" % (self.episodecount))

        if len(self.trans_mem) > self.batch_size*10:

            self.update_count += 1

            minibatch = self.sample(self.trans_mem, self.priority_mem, self.batch_size)
            batchify = lambda x: list(x) * self.weight_num
            state_batch = batchify(map(lambda x: x.s, minibatch))
            action_batch = batchify(map(lambda x: LongTensor([x.a]), minibatch))
            reward_batch = batchify(map(lambda x: x.r.unsqueeze(0), minibatch))
            next_state_batch = batchify(map(lambda x: x.s_, minibatch))
            terminal_batch = batchify(map(lambda x: x.d, minibatch))
            mask_batch = batchify(map(lambda x: x.ms.unsqueeze(0), minibatch))
            next_mask_batch = batchify(map(lambda x: x.ms_.unsqueeze(0), minibatch))

            w_batch = np.random.randn(self.weight_num, self.model_.reward_size)
            w_batch = np.abs(w_batch) / \
                      np.linalg.norm(w_batch, ord=1, axis=1, keepdims=True)
            w_batch = torch.from_numpy(w_batch.repeat(self.batch_size, axis=0)).type(FloatTensor)

            if self.algorithm == 'naive':
                __, Q = self.model_(Variable(torch.cat(state_batch, dim=0)),
                                    Variable(w_batch),
                                    Variable(torch.cat(mask_batch, dim=0)))
                # detach since we don't want gradients to propagate
                # HQ, _    = self.model_(Variable(torch.cat(next_state_batch, dim=0), volatile=True),
                #                     Variable(w_batch, volatile=True))
                _, DQ = self.model(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                                   Variable(w_batch, requires_grad=False),
                                   Variable(torch.cat(next_mask_batch, dim=0), requires_grad=False))
                _, act = self.model_(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                                     Variable(w_batch, requires_grad=False),
                                     Variable(torch.cat(next_mask_batch, dim=0), requires_grad=False))[1].max(1)
                HQ = DQ.gather(1, act.unsqueeze(dim=1)).squeeze()

                w_reward_batch = torch.bmm(w_batch.unsqueeze(1),
                                           torch.cat(reward_batch, dim=0).unsqueeze(2)
                                           ).squeeze()

                nontmlmask = self.nontmlinds(terminal_batch)
                with torch.no_grad():
                    Tau_Q = Variable(torch.zeros(self.batch_size * self.weight_num).type(FloatTensor))
                    Tau_Q[nontmlmask] = self.gamma * HQ[nontmlmask]
                    Tau_Q += Variable(w_reward_batch)

                actions = Variable(torch.cat(action_batch, dim=0))

                # Compute Huber loss
                loss = F.smooth_l1_loss(Q.gather(1, actions.unsqueeze(dim=1)), Tau_Q.unsqueeze(dim=1))

            elif self.algorithm == 'envelope':
                action_size = self.model_.action_size
                reward_size = self.model_.reward_size
                __, Q = self.model_(Variable(torch.cat(state_batch, dim=0)),
                                    Variable(w_batch),
                                    w_num=self.weight_num,
                                    execmask=Variable(torch.cat(mask_batch, dim=0)))

                # detach since we don't want gradients to propagate
                # HQ, _    = self.model_(Variable(torch.cat(next_state_batch, dim=0), volatile=True),
                #                     Variable(w_batch, volatile=True), w_num=self.weight_num)
                _, DQ = self.model(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                                   Variable(w_batch, requires_grad=False),
                                   execmask=Variable(torch.cat(next_mask_batch, dim=0), requires_grad=False))
                w_ext = w_batch.unsqueeze(2).repeat(1, action_size, 1)
                w_ext = w_ext.view(-1, self.model.reward_size)
                _, tmpQ = self.model_(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                                      Variable(w_batch, requires_grad=False),
                                      execmask=Variable(torch.cat(next_mask_batch, dim=0), requires_grad=False))

                tmpQ = tmpQ.view(-1, reward_size)
                # print(torch.bmm(w_ext.unsqueeze(1),
                #               tmpQ.data.unsqueeze(2)).view(-1, action_size))
                act = torch.bmm(Variable(w_ext.unsqueeze(1), requires_grad=False),
                                tmpQ.unsqueeze(2)).view(-1, action_size).max(1)[1]

                HQ = DQ.gather(1, act.view(-1, 1, 1).expand(DQ.size(0), 1, DQ.size(2))).squeeze()

                nontmlmask = self.nontmlinds(terminal_batch)
                with torch.no_grad():
                    Tau_Q = Variable(torch.zeros(self.batch_size * self.weight_num,
                                                 reward_size).type(FloatTensor))
                    Tau_Q[nontmlmask] = self.gamma * HQ[nontmlmask]
                    # Tau_Q.volatile = False
                    Tau_Q += Variable(torch.cat(reward_batch, dim=0))

                actions = Variable(torch.cat(action_batch, dim=0))

                Q = Q.gather(1, actions.view(-1, 1, 1).expand(Q.size(0), 1, Q.size(2))
                             ).view(-1, reward_size)
                Tau_Q = Tau_Q.view(-1, reward_size)

                wQ = torch.bmm(Variable(w_batch.unsqueeze(1)),
                               Q.unsqueeze(2)).squeeze()

                wTQ = torch.bmm(Variable(w_batch.unsqueeze(1)),
                                Tau_Q.unsqueeze(2)).squeeze()

                # loss = F.mse_loss(Q.view(-1), Tau_Q.view(-1))
                # print self.beta
                loss = self.beta * F.mse_loss(wQ.view(-1), wTQ.view(-1))
                loss += (1 - self.beta) * F.mse_loss(Q.view(-1), Tau_Q.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model_.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            if self.update_count % self.update_freq == 0:
                self.model.load_state_dict(self.model_.state_dict())

            self.monitor.update(self.episodecount, loss=loss.data)

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
            torch.save(self.model, "{}.{}.pkl".format(self.out_policy_file, self.algorithm))

    def loadPolicy(self, filename):
        """
        load model and replay buffer
        """
        # load models
        self.model_ = torch.load("{}.{}.pkl".format(filename, self.algorithm))
        self.model = copy.deepcopy(self.model_)

    def restart(self):
        self.summaryAct = None
        self.lastSystemAction = None
        self.prevbelief = None
        self.actToBeRecorded = None
        self.w_kept = None
        if self.epsilon_decay:
            self.epsilon -= self.epsilon_delta
        if self.homotopy:
            self.beta += self.beta_delta
            self.beta_delta = (self.beta - self.beta_init) * self.beta_expbase + self.beta_init - self.beta

# END OF FILE
