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
ConfusionModel.py - handcrafted SemI error creator 
===================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.DiaAct` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger`

************************

''' 

__author__ = "cued_dialogue_systems_group"
import copy

from utils import Settings
from utils import DiaAct
from ontology import Ontology
from utils import ContextLogger
import numpy as np
logger = ContextLogger.getLogger('')


class EMConfusionModel(object):
    '''Base class for EMRandomConfusionModel. 

        .. Note:: 
            Used through derived class only. 
    '''
    def create_wrong_hyp(self, a_u):
        '''Create a wrong hypothesis for a_u

        :param a_u: of :class:`DiaAct`
        :type a_u: instance
        :returns: (instance) of :class:`DiaAct` - modified input act
        '''
        confact_is_same = True
        num_attempts = 0
        max_num_attempts = 25
        conf_act = None
        while confact_is_same and num_attempts < max_num_attempts:
            conf_act = self.confuse_hyp(a_u)
            confact_is_same = (conf_act == a_u)
            if conf_act.act == 'bye':
                confact_is_same = True # hack to avoid the system finishing the dialogue after a bye confusion
            num_attempts += 1

        if num_attempts == max_num_attempts:
            logger.warning("Confused act same after %d attempts: return null() instead." % max_num_attempts)
            #return DiaAct.DiaAct('null()')
            return a_u

        return conf_act


class EMRandomConfusionModel(EMConfusionModel):
    '''Derived class from :class:`EMConfusionModel`.

    :param None:
    '''

    def __init__(self, domainString):
        self.domainString = domainString

        self.CONFUSE_TYPE = 0.2
        self.CONFUSE_SLOT = 0.3
        self.CONFUSE_VALUE = 0.5
        self.newUserActs = ['hello',
                            'thankyou',
                            'ack',
                            'bye',
                            'inform',
                            'request',
                            'reqalts',
                            'reqmore',
                            'confirm',
                            'affirm',
                            'negate',
                            'deny',
                            'repeat',
                            'null']
        self.nNewUserActs = len(self.newUserActs)

    def confuse_hyp(self, a_u):
        '''Randomly confuse the act type, slot or value.

        :param a_u: of :class:`DiaAct`
        :type a_u: instance
        :returns: (instance) of :class:`DiaAct` - modified input act
        '''
        wHyp = copy.deepcopy(a_u)

        # Identify if this diaact type takes 0, 1, or 2 arguments
        nSlotVal = wHyp.getDiaItemFormat()

        # Make a choice to confuse the type, slot or value
        choice = Settings.random.choice([0, 1, 2], p=[self.CONFUSE_TYPE, self.CONFUSE_SLOT, self.CONFUSE_VALUE])
        choice = min(choice, nSlotVal)

        if choice == 0:
            wHyp = self._confuse_type(wHyp)
        elif choice == 1:
            wHyp = self._confuse_slot(wHyp)
        elif choice == 2:
            wHyp = self._confuse_value(wHyp)
        else:
            logger.error('Invalid choice ' + str(choice))

        return wHyp

    def _confuse_dia_act_type(self, oldtype):
        '''
        Randomly select a dialogue act type different from oldtype.
        '''
        acttypeset = copy.copy(self.newUserActs)
        acttypeset.remove(oldtype)
        return Settings.random.choice(acttypeset)

    def _confuse_slot_name(self, old_name):
        '''
        Randomly select a slot name that is different from the given old_name
        '''
        slots = Ontology.global_ontology.get_requestable_slots(self.domainString)
        if old_name in slots:
            slots.remove(old_name)
        # if old_name not in slots:
        #     logger.error('Slot "%s" is not found in ontology.' % old_name)

        return Settings.random.choice(slots)

    def _get_confused_value_for_slot(self, slot, old_val):
        '''
        Randomly select a slot value for the given slot s different from old_val.
        '''
        return Ontology.global_ontology.getRandomValueForSlot(self.domainString, slot=slot, notthese=[old_val])

    def _confuse_type(self, hyp):
        '''
        Create a wrong hypothesis, where the dialogue act type is different.
        '''
        hyp.items = []
        hyp.act = self._confuse_dia_act_type(hyp.act)
        item_format = DiaAct.actTypeToItemFormat[hyp.act]
        if item_format == 0:
            return hyp
        elif item_format == 1:
            new_slot_name = Ontology.global_ontology.get_random_slot_name(self.domainString)
            hyp.append(new_slot_name, None)
        elif item_format == 2:
            new_slot_name = Ontology.global_ontology.get_random_slot_name(self.domainString)
            assert new_slot_name is not None
            new_slot_val = Ontology.global_ontology.getRandomValueForSlot(self.domainString, slot=new_slot_name)
            hyp.append(new_slot_name, new_slot_val)
        # TODO: If item_format is 3, it doesn't confuse slot-values.
        # This might be a bug in the original implementation.
        return hyp

    def _confuse_slot(self, hyp):
        '''
        Create a wrong hypothesis, where the slot names are different.
        '''
        for dip in hyp.items:
            # If the slot is empty, just break
            if dip.slot is None:
                break

            slot = dip.slot
            if slot == 'more':
                break

            dip.slot = self._confuse_slot_name(slot)
            if dip.val is not None:
                dip.val = Ontology.global_ontology.getRandomValueForSlot(self.domainString, slot=dip.slot)

        return hyp

    def _confuse_value(self, a_u):
        '''
        Create a wrong hypothesis, where one slot value is different.
        '''
        rand = Settings.random.randint(len(a_u.items))
        a_u_i = a_u.items[rand]

        if a_u_i.slot is not None and a_u_i.val is not None:
            a_u.items[rand].val = self._get_confused_value_for_slot(a_u_i.slot, a_u_i.val)

        return a_u

class EMLevenshteinConfusionModel(EMConfusionModel):
    '''Derived class from :class:`EMConfusionModel`.

    :param None:
    '''

    def __init__(self, domainString):
        self.domainString = domainString

        self.CONFUSE_TYPE = 0.2
        self.CONFUSE_SLOT = 0.3
        self.CONFUSE_VALUE = 0.5
        self.len_confusion_list = 6
        self.newUserActs = ['hello',
                            'thankyou',
                            'ack',
                            'bye',
                            'inform',
                            'request',
                            'reqalts',
                            'reqmore',
                            'confirm',
                            'affirm',
                            'negate',
                            'deny',
                            'repeat',
                            'null']
        self.nNewUserActs = len(self.newUserActs)
        self.type_confusions = self.get_confusion_distributions(self.newUserActs, offset=0.15)
        self.slot_confusions = self.get_confusion_distributions(Ontology.global_ontology.get_requestable_slots(self.domainString), offset=0.15)
        self.slot_value_confusions = {}
        for slot in Ontology.global_ontology.get_system_requestable_slots(self.domainString) + [unicode('name')]:
            self.slot_value_confusions[slot] = self.get_confusion_distributions(
                Ontology.global_ontology.get_informable_slot_values(self.domainString, slot) + [unicode('dontcare')], offset=0.15)

    def get_confusion_distributions(self, word_list, offset=0.15):
        '''

        :param word_list: The list of words to be confused
        :param offset: Distribution softening factor, the largest the softer the distribution will be
        :return: dictionary
        '''
        wlist = list(word_list)
        Settings.random.shuffle(wlist)
        distributions = {}
        distances = [[self.levenshteinDistance(w1,w2) for w1 in wlist] for w2 in wlist]
        for i in range(len(wlist)):
            word = wlist[i]
            distributions[word] = {}
            sorted_indexes = np.argsort(distances[i])[1:self.len_confusion_list+1]
            sorted_wordlist = np.array(wlist)[sorted_indexes]
            distribution = np.array(distances[i])[sorted_indexes]
            distribution = 1./distribution
            distribution /= sum(distribution)
            distribution += offset
            distribution /= sum(distribution)
            distributions[word]['wlist'] = sorted_wordlist
            distributions[word]['dist'] = distribution
        return distributions

    def confuse_hyp(self, a_u):
        '''Randomly confuse the act type, slot or value.

        :param a_u: of :class:`DiaAct`
        :type a_u: instance
        :returns: (instance) of :class:`DiaAct` - modified input act
        '''
        wHyp = copy.deepcopy(a_u)

        # Identify if this diaact type takes 0, 1, or 2 arguments
        nSlotVal = wHyp.getDiaItemFormat()

        # Make a choice to confuse the type, slot or value
        choice = Settings.random.choice([0, 1, 2], p=[self.CONFUSE_TYPE, self.CONFUSE_SLOT, self.CONFUSE_VALUE])
        choice = min(choice, nSlotVal)

        if choice == 0:
            wHyp = self._confuse_type(wHyp)
        elif choice == 1:
            wHyp = self._confuse_slot(wHyp)
        elif choice == 2:
            wHyp = self._confuse_value(wHyp)
        else:
            logger.error('Invalid choice ' + str(choice))

        return wHyp

    def _confuse_dia_act_type(self, oldtype):
        '''
        Select a dialogue act type different from oldtype.
        '''
        return Settings.random.choice(self.type_confusions[oldtype]['wlist'], p=self.type_confusions[oldtype]['dist'])


    def _confuse_slot_name(self, old_name):
        '''
        Randomly select a slot name that is different from the given old_name
        '''
        return Settings.random.choice(self.slot_confusions[old_name]['wlist'], p=self.slot_confusions[old_name]['dist'])

    def _get_confused_value_for_slot(self, slot, old_val):
        '''
        Randomly select a slot value for the given slot s different from old_val.
        '''
        if slot not in self.slot_value_confusions.keys():
            return Ontology.global_ontology.getRandomValueForSlot(self.domainString, slot=slot, notthese=[old_val])
        elif old_val not in self.slot_value_confusions[slot]:
            return Ontology.global_ontology.getRandomValueForSlot(self.domainString, slot=slot, notthese=[old_val])
        else:
            return Settings.random.choice(self.slot_value_confusions[slot][old_val]['wlist'], p=self.slot_value_confusions[slot][old_val]['dist'])

    def _confuse_type(self, hyp):
        '''
        Create a wrong hypothesis, where the dialogue act type is different.
        '''
        hyp.items = []
        hyp.act = self._confuse_dia_act_type(hyp.act)
        item_format = DiaAct.actTypeToItemFormat[hyp.act]
        if item_format == 0:
            return hyp
        elif item_format == 1:
            new_slot_name = Ontology.global_ontology.get_random_slot_name(self.domainString)
            hyp.append(new_slot_name, None)
        elif item_format == 2:
            new_slot_name = Ontology.global_ontology.get_random_slot_name(self.domainString)
            assert new_slot_name is not None
            new_slot_val = Ontology.global_ontology.getRandomValueForSlot(self.domainString, slot=new_slot_name)
            hyp.append(new_slot_name, new_slot_val)
        # TODO: If item_format is 3, it doesn't confuse slot-values.
        # This might be a bug in the original implementation.
        return hyp

    def _confuse_slot(self, hyp):
        '''
        Create a wrong hypothesis, where the slot names are different.
        '''
        for dip in hyp.items:
            # If the slot is empty, just break
            if dip.slot is None:
                break

            slot = dip.slot
            if slot == 'more' or slot == 'type':
                break

            dip.slot = self._confuse_slot_name(slot)
            if dip.val is not None:
                dip.val = Ontology.global_ontology.getRandomValueForSlot(self.domainString, slot=dip.slot)

        return hyp

    def _confuse_value(self, a_u):
        '''
        Create a wrong hypothesis, where one slot value is different.
        '''
        rand = Settings.random.randint(len(a_u.items))
        a_u_i = a_u.items[rand]

        if a_u_i.slot is not None and a_u_i.val is not None and a_u_i.slot != 'type':
            a_u.items[rand].val = self._get_confused_value_for_slot(a_u_i.slot, a_u_i.val)

        return a_u

    def levenshteinDistance(self, s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

#END OF FILE
