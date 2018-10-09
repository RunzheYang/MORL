###############################################################################
# PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015 - 2018
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
FeudalPolicy.py - Feudal policy based on the work presente in  https://arxiv.org/pdf/1803.03232.pdf
============================================

Copyright CUED Dialogue Systems Group 2015 - 2018


**Relevant Config variables** [Default and possible values]::

    [policy]
    policytype = feudal

    [feudalpolicy]
    features = dip/learned/rnn
    sortbelief = True/False
    si_enc_size = 25
    dropout_rate = 0
    si_policy_type = dqn
    sd_policy_type = dqn
    master_policy_type = dqn
    actfreq_ds = True/False



.. seealso:: CUED Imports/Dependencies:

    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger`
    import :mod:`policy.feudalRL.DIP_parametrisation`
    import :mod:`policy.feudalRL.FeudalDQNPolicy`
    import :mod:`policy.feudalRL.FeudalBBQNPolicy`
    import :mod:`policy.feudalRL.FeudalENACPolicy`
    import :mod:`policy.feudalRL.FeudalACERPolicy`
    import :mod:`policy.feudalRL.feudalUtils`


************************

'''

__author__ = "cued_dialogue_systems_group"

import sys
import numpy as np
import utils
from utils.Settings import config as cfg
from utils import ContextLogger, DiaAct

import ontology.FlatOntologyManager as FlatOnt

import Policy
import SummaryAction
from policy.feudalRL.DIP_parametrisation import DIP_state, padded_state
from policy.feudalRL.FeudalDQNPolicy import FeudalDQNPolicy
try:
	from policy.feudalRL.FeudalBBQNPolicy import FeudalBBQNPolicy
	from policy.feudalRL.FeudalENACPolicy import FeudalENACPolicy
	from policy.feudalRL.FeudalACERPolicy import FeudalACERPolicy
except:
	pass
from policy.feudalRL.feudalUtils import get_feudal_masks

logger = utils.ContextLogger.getLogger('')


class FeudalPolicy(Policy.Policy):
    '''Derived from :class:`Policy`
    '''

    def __init__(self, in_policy_file, out_policy_file, domainString='CamRestaurants', is_training=False):
        super(FeudalPolicy, self).__init__(domainString, is_training)

        self.domainString = domainString
        self.domainUtil = FlatOnt.FlatDomainOntology(self.domainString)
        self.in_policy_file = in_policy_file
        self.out_policy_file = out_policy_file
        self.is_training = is_training
        self.prev_state_check = None

        #feudalRL variables
        self.prev_sub_policy = None
        self.prev_master_act = None
        self.prev_master_belief = None
        self.prev_child_act = None
        self.prev_child_belief = None

        self.action_freq = np.zeros(len(self.actions.action_names))

        self.master_dec_count = np.array([0.,0.])
        self.gi_dec_inrow = 0


        self.features = 'dip'
        if cfg.has_option('feudalpolicy', 'features'):
            self.features = cfg.get('feudalpolicy', 'features')
        self.si_policy_type = 'dqn'
        if cfg.has_option('feudalpolicy', 'si_policy_type'):
            self.si_policy_type = cfg.get('feudalpolicy', 'si_policy_type')
        self.sd_policy_type = 'dqn'
        if cfg.has_option('feudalpolicy', 'sd_policy_type'):
            self.sd_policy_type = cfg.get('feudalpolicy', 'sd_policy_type')
        self.master_policy_type = self.si_policy_type
        if cfg.has_option('feudalpolicy', 'master_policy_type'):
            self.master_policy_type = cfg.get('feudalpolicy', 'master_policy_type')
        self.sample_master = False
        if cfg.has_option('feudalpolicy', 'sample_master'):
            self.sample_master = cfg.getboolean('feudalpolicy', 'sample_master')
        self.correct_master = False
        if cfg.has_option('feudalpolicy', 'correct_master'):
            self.correct_master = cfg.getboolean('feudalpolicy', 'correct_master')
        self.use_bye = False
        if cfg.has_option('feudalpolicy', 'use_bye'):
            self.use_bye = cfg.getboolean('feudalpolicy', 'use_bye')
        self.reqmore_in_si = True
        if cfg.has_option('feudalpolicy', 'reqmore_in_si'):
            self.reqmore_in_si = cfg.getboolean('feudalpolicy', 'reqmore_in_si')
        self.correction_factor = 0
        if cfg.has_option('feudalpolicy', 'correction_factor'):
            self.correction_factor = cfg.getfloat('feudalpolicy', 'correction_factor')
        self.actfreq_ds = False
        if cfg.has_option('feudalpolicy', 'actfreq_ds'):
            self.actfreq_ds = cfg.getboolean('feudalpolicy', 'actfreq_ds')

        # parameter settings

        self.randomseed = 1234
        if cfg.has_option('GENERAL', 'seed'):
            self.randomseed = cfg.getint('GENERAL', 'seed')

        # Create the feudal structure (including feudal masks)

        self.summaryaction = SummaryAction.SummaryAction(domainString)
        self.full_action_list = self.summaryaction.action_names

        self.master_actions = ['give_info', 'request_info', 'pass']

        self.slot_independent_actions = ["inform",
                                         "inform_byname",
                                         "inform_alternatives"
                                        ]
        if self.reqmore_in_si:
            self.slot_independent_actions.append("reqmore")
        if self.use_bye:
            self.slot_independent_actions.append('bye')
        self.slot_independent_actions.append('pass')

        self.slot_specific_actions = ["request",
                                      "confirm",
                                      "select"]
        #if self.reqmore_in_sd is True:
        #    self.slot_specific_actions.append("reqmore")
        self.slot_specific_actions.append('pass')

        self.master_freq = np.zeros(len(self.master_actions))
        self.si_freq = np.zeros(len(self.slot_independent_actions))
        self.sd_freq = np.zeros(len(self.slot_specific_actions))

        # master policy
        if self.master_policy_type == 'acer':
            self.master_policy = FeudalACERPolicy(self._modify_policyfile('master', in_policy_file),
                                                  self._modify_policyfile('master', out_policy_file),
                                                  domainString=self.domainString, is_training=self.is_training,
                                                  action_names=['give_info', 'request_info', 'pass'],
                                                  slot='si')  # pass is always masked, but its needed for implementation
        elif self.master_policy_type == 'enac':
            self.master_policy = FeudalENACPolicy(self._modify_policyfile('master', in_policy_file),
                                                  self._modify_policyfile('master', out_policy_file),
                                                  domainString=self.domainString, is_training=self.is_training,
                                                  action_names=['give_info', 'request_info', 'pass'],
                                                  slot='si')  # pass is always masked, but its needed for implementation
        elif self.master_policy_type == 'bbqn':
            self.master_policy = FeudalBBQNPolicy(self._modify_policyfile('master', in_policy_file),
                                                  self._modify_policyfile('master', out_policy_file),
                                                  domainString=self.domainString, is_training=self.is_training,
                                                  action_names=['give_info', 'request_info', 'pass'],
                                                  slot='si')  # pass is always masked, but its needed for implementation
        else:
            self.master_policy = FeudalDQNPolicy(self._modify_policyfile('master', in_policy_file),
                                                  self._modify_policyfile('master', out_policy_file),
                                                  domainString=self.domainString, is_training=self.is_training,
                                                  action_names=['give_info', 'request_info', 'pass'],
                                                  slot='si')  # pass is always masked, but its needed for implementation
        # si policy
        if self.si_policy_type == 'acer':
            self.give_info_policy = FeudalACERPolicy(self._modify_policyfile('gi', in_policy_file),
                                                     self._modify_policyfile('gi', out_policy_file),
                                                     domainString=self.domainString, is_training=self.is_training,
                                                     action_names=self.slot_independent_actions, slot='si')
        elif self.si_policy_type == 'enac':
            self.give_info_policy = FeudalENACPolicy(self._modify_policyfile('gi', in_policy_file),
                                                    self._modify_policyfile('gi', out_policy_file),
                                                    domainString=self.domainString, is_training=self.is_training,
                                                    action_names=self.slot_independent_actions, slot='si')
        elif self.si_policy_type == 'bbqn':
            self.give_info_policy = FeudalBBQNPolicy(self._modify_policyfile('gi', in_policy_file),
                                                    self._modify_policyfile('gi', out_policy_file),
                                                    domainString=self.domainString, is_training=self.is_training,
                                                    action_names=self.slot_independent_actions, slot='si')
        else:
            self.give_info_policy = FeudalDQNPolicy(self._modify_policyfile('gi', in_policy_file),
                                                    self._modify_policyfile('gi', out_policy_file),
                                                     domainString=self.domainString, is_training=self.is_training,
                                                     action_names=self.slot_independent_actions, slot='si')

        # sd policies
        if self.sd_policy_type == 'acer':
            self.request_info_policy = FeudalACERPolicy(self._modify_policyfile('ri', in_policy_file),
                                                   self._modify_policyfile('ri', out_policy_file),
                                                 domainString=self.domainString, is_training=self.is_training,
                                                 action_names=self.slot_specific_actions, slot='sd')
        elif self.sd_policy_type == 'bbqn':
            self.request_info_policy = FeudalBBQNPolicy(self._modify_policyfile('ri', in_policy_file),
                                                   self._modify_policyfile('ri', out_policy_file),
                                                 domainString=self.domainString, is_training=self.is_training,
                                                 action_names=self.slot_specific_actions, slot='sd')
        else:
            self.request_info_policy = FeudalDQNPolicy(self._modify_policyfile('ri', in_policy_file),
                                                       self._modify_policyfile('ri', out_policy_file),
                                                       domainString=self.domainString, is_training=self.is_training,
                                                       action_names=self.slot_specific_actions, slot='sd')

    def _modify_policyfile(self, mod, policyfile):
        pf_split = policyfile.split('/')
        pf_split[-1] = mod + '_' + pf_split[-1]
        return '/'.join(pf_split)

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
        self.master_policy.record(reward, domainInControl=self.domainString, state=self.prev_master_belief, action=self.prev_master_act)
        if self.prev_sub_policy == 0:
            self.give_info_policy.record(reward, domainInControl=self.domainString, state=self.prev_child_belief, action=self.prev_child_act)
            self.request_info_policy.record(reward, domainInControl=self.domainString, state=self.prev_child_belief,
                                            action=len(self.slot_specific_actions)-1)
        elif self.prev_sub_policy == 1:
            self.request_info_policy.record(reward, domainInControl=self.domainString, state=self.prev_child_belief, action=self.prev_child_act)
            self.give_info_policy.record(reward, domainInControl=self.domainString, state=self.prev_child_belief,
                                            action=len(self.slot_independent_actions)-1)

    def finalizeRecord(self, reward, domainInControl=None):
        if domainInControl is None:
            domainInControl = self.domainString
        self.master_policy.finalizeRecord(reward)
        self.give_info_policy.finalizeRecord(reward)
        self.request_info_policy.finalizeRecord(reward)

    def convertStateAction(self, state, action):
        pass
        #this aparently is not necesary

    def nextAction(self, beliefstate):
        '''
        select next action

        :param beliefstate:
        :returns: (int) next summary action
        '''
        # compute main belief
        af = None
        if self.actfreq_ds:
            #af = 1./(1 + self.action_freq)
            af = 1./(1 + np.concatenate((self.si_freq, self.sd_freq)))
        if self.features == 'learned' or self.features == 'rnn':
            dipstate = padded_state(beliefstate, domainString=self.domainString, action_freq=af)
        else:
            dipstate = DIP_state(beliefstate,domainString=self.domainString, action_freq=af)
        dipstatevec = dipstate.get_beliefStateVec('general')
        # Make decision on main policy
        master_Q_values = self.master_policy.nextAction(dipstatevec)
        non_exec = self.summaryaction.getNonExecutable(beliefstate.domainStates[beliefstate.currentdomain], self.lastSystemAction)
        masks = get_feudal_masks(non_exec, dipstate.slots, self.slot_independent_actions, self.slot_specific_actions)
        master_Q_values = np.add(master_Q_values, masks['master'])
        if self.is_training and self.correction_factor != 0:
            correction = (1-self.master_freq/sum(self.master_freq))
            master_Q_values *= correction
        if self.sample_master is True and self.is_training is False:
            probs = master_Q_values[:-1]
            if np.any([x for x in probs if x<0]):
                probs[[x < 0 for x in probs]] = 0
            probs /= sum(probs)
            master_decision = np.random.choice([0,1],p=probs)
            #print master_decision
        else:
            master_decision = np.argmax(master_Q_values)
        if master_decision == 0 and self.gi_dec_inrow == 4 and self.correct_master and not self.is_training:
            master_decision = 1
        self.master_freq[master_decision] += 1
        if not self.is_training:
            self.master_dec_count[master_decision] += 1
            if np.sum(self.master_dec_count) % 1000 == 0:
                logger.results('master action frequencies = {}'.format(list(self.master_dec_count)/np.sum(self.master_dec_count))) #TODO: change to debug
        #print 'master Q:', master_Q_values, 'master decision:', master_decision
        self.prev_master_act = master_decision
        self.prev_master_belief = dipstatevec
        if master_decision == 0:
            self.gi_dec_inrow += 1.
            # drop to give_info policy
            self.prev_sub_policy = 0
            child_Q_values = self.give_info_policy.nextAction(dipstatevec)
            child_Q_values = np.add(child_Q_values, masks['give_info'])
            child_decision = np.argmax(child_Q_values)
            summaryAct = self.slot_independent_actions[child_decision]
            self.prev_child_act = child_decision
            self.prev_child_belief = dipstatevec
            #print 'give info Q:', child_Q_values, 'give info decision:', summaryAct
            self.si_freq[child_decision] += 1

        elif master_decision == 1:
            self.gi_dec_inrow = 0
            # drop to request_info policy
            self.prev_sub_policy = 1
            slot_Qs = {}
            best_action = ('slot', 'action', -np.inf)
            for slot in dipstate.slots:
                dipstatevec = dipstate.get_beliefStateVec(slot)
                slot_Qs[slot] = self.request_info_policy.nextAction(dipstatevec)
                slot_Qs[slot] = np.add(slot_Qs[slot], masks['req_info'][slot])
                slot_max_Q = np.max(slot_Qs[slot])
                if slot_max_Q > best_action[2]:
                    best_action = (slot, np.argmax(slot_Qs[slot]), slot_max_Q)
            summaryAct = self.slot_specific_actions[best_action[1]] + '_' + best_action[0]
            if 'reqmore' in summaryAct:
                summaryAct = 'reqmore'
            self.prev_child_act = best_action[1]
            self.prev_child_belief = dipstate.get_beliefStateVec(best_action[0])
            self.sd_freq[best_action[1]] += 1
            #print 'req info Q:', [slot_Qs[s] for s in slot_Qs], 'req info decision:', summaryAct

        self.action_freq[self.actions.action_names.index(summaryAct)] += 1
        #print  1./(1+self.action_freq)
        beliefstate = beliefstate.getDomainState(self.domainUtil.domainString)
        masterAct = self.summaryaction.Convert(beliefstate, summaryAct, self.lastSystemAction)
        nextaIdex = self.full_action_list.index(summaryAct)
        return masterAct, nextaIdex

    def get_feudal_masks(self, belief, last_sys_act, slots):
        belief = belief.domainStates[belief.currentdomain]
        non_exec = self.summaryaction.getNonExecutable(belief, last_sys_act)
        feudal_masks = {'req_info':{}, 'give_info':None, 'master':None}
        give_info_masks = np.zeros(len(self.slot_independent_actions))
        give_info_masks[-1] = -sys.maxint
        for i, action in enumerate(self.slot_independent_actions):
            if action in non_exec:
                give_info_masks[i] = -sys.maxint
        feudal_masks['give_info'] = give_info_masks
        for slot in slots:
            feudal_masks['req_info'][slot] = np.zeros(len(self.slot_specific_actions))
            feudal_masks['req_info'][slot][-1] = -sys.maxint
            for i, action in enumerate(self.slot_specific_actions):
                if action+'_'+slot in non_exec:
                    feudal_masks['req_info'][slot][i] = -sys.maxint
        master_masks = np.zeros(3)
        master_masks[:] = -sys.maxint
        if 0 in give_info_masks:
            master_masks[0] = 0
        for slot in slots:
            if 0 in feudal_masks['req_info'][slot]:
                master_masks[1] = 0
        feudal_masks['master'] = master_masks
        #print(non_exec)
        #print(feudal_masks)
        return feudal_masks

    def train(self):
        '''
        call this function when the episode ends
        '''
        #just train each sub-policy
        #print 'train master'
        self.master_policy.train()
        #print 'train gi'
        self.give_info_policy.train()
        #print 'train ri'
        self.request_info_policy.train()

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
        # just save each sub-policy
        self.master_policy.savePolicyInc()
        self.give_info_policy.savePolicyInc()
        self.request_info_policy.savePolicyInc()

    def loadPolicy(self, filename):
        """
        load model and replay buffer
        """
        # load policy models one by one
        pass

    def restart(self):
        self.summaryAct = None
        self.lastSystemAction = None
        self.prevbelief = None
        self.actToBeRecorded = None
        self.master_policy.restart()
        self.give_info_policy.restart()
        self.request_info_policy.restart()
        self.master_freq = np.zeros(len(self.master_actions))
        self.si_freq = np.zeros(len(self.slot_independent_actions))
        self.sd_freq = np.zeros(len(self.slot_specific_actions))
        self.action_freq = np.zeros(len(self.actions.action_names))

# END OF FILE
