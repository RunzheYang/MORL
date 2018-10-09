'''
Class to convert belief states into DIP parametrisations
'''

import numpy as np
import copy
from itertools import product
from scipy.stats import entropy

from policy.Policy import Policy, Action, State, TerminalAction, TerminalState
from ontology import Ontology
from utils import Settings, ContextLogger, DialogueState
logger = ContextLogger.getLogger('')

class DIP_state(State):
    def __init__(self, belief, domainString=None, action_freq=None):
        #params
        self.domainString = domainString
        self.N_bins = 10
        self.slots = list(Ontology.global_ontology.get_informable_slots(domainString))
        if 'price' in self.slots:
            self.slots.remove('price') #remove price from SFR ont, its not used

        if 'name' in self.slots:
            self.slots.remove('name')
        self.DIP_state = {'general':None, 'joint':None}
        for slot in self.slots:
            self.DIP_state[slot]=None

        # convert belief state into DIP params
        if action_freq is not None:
            self.DIP_state['general'] = np.concatenate((action_freq,self.convert_general_b(belief)))
        else:
            self.DIP_state['general'] = self.convert_general_b(belief)
        self.DIP_state['joint'] = self.convert_joint_slot_b(belief)
        for slot in self.slots:
            self.DIP_state[slot] = self.convert_slot_b(belief, slot)

        # create DIP vector and masks
        self.get_DIP_vector()
        self.beliefStateVec = None #for compatibility with GP sarsa implementation

    def get_DIP_vector(self):
        """
        convert the DIP state into a numpy vector and a set of masks per slot
        :return:
        """
        pad_v = np.zeros(len(self.DIP_state[self.slots[0]]))
        slot_len = len(pad_v)
        general_len = len(self.DIP_state['general']) + len(self.DIP_state['joint'])
        pad_v[0] = 1.
        self.DIP_vector = [pad_v]
        self.DIP_masks = {}
        mask_template = [False] * (slot_len * (len(self.slots) + 1)) + [True] * general_len
        i = 1
        for slot in self.slots:
            self.DIP_vector.append(self.DIP_state[slot])
            self.DIP_masks[slot] = np.array(mask_template)
            self.DIP_masks[slot][slot_len*i:slot_len*(i+1)] = True
            i += 1
        self.DIP_vector.append(self.DIP_state['general'])
        self.DIP_vector.append(self.DIP_state['joint'])
        self.DIP_masks['general'] = np.array(mask_template)
        self.DIP_masks['general'][:slot_len] = True

        self.DIP_vector = np.concatenate(self.DIP_vector)

    def get_beliefStateVec(self, slot):
        return self.DIP_vector[self.DIP_masks[slot]]

    def get_DIP_state(self, slot):
        return np.array([self.DIP_state['general'] + self.DIP_state['joint'] + self.DIP_state[slot]])

    def get_full_DIP_state(self):
        full_slot_bstate = []
        for slot in self.slots:
            full_slot_bstate += self.DIP_state[slot]
        full_DIP_state = np.array([full_slot_bstate + self.DIP_state['general'] + self.DIP_state['joint']])
        DIP_mask = [True]*(len(self.DIP_state['general']) + len(self.DIP_state['joint'])) + [False] * len(full_slot_bstate)
        return full_DIP_state, DIP_mask

    def convert_general_b(self, belief):
        """
        Extracts from the belief state the DIP vector corresponding to the general features (e.g. method, user act...)
        :param belief: The full belief state
        :return: The DIP general vector
        """
        if type(belief) == DialogueState.DialogueState:
            belief = belief.domainStates[belief.currentdomain]

        dial_act = belief['beliefs']['discourseAct'].values()

        requested = self._get_DIP_requested_vector(belief)
        method = belief['beliefs']['method'].values()
        features = [int(belief['features']['offerHappened']), int(belief['features']['lastActionInformNone']), int(bool(belief['features']['lastInformedVenue']))]
        discriminable = [int(x) for x in belief['features']['inform_info']]
        slot_n = 1/len(self.slots)
        val_n = []
        for slot in self.slots:
            val_n.append(len(Ontology.global_ontology.get_informable_slot_values(self.domainString, slot)))
        avg_value_n = 1/np.mean(val_n)


        return dial_act + requested + method + features + discriminable + [slot_n, avg_value_n]


    def _get_DIP_requested_vector(self, belief):
        n_requested = sum([x>0.5 for x in belief['beliefs']['requested'].values()])
        ret_vec = [0] * 5
        if n_requested > 4:
            n_requested = 4
        ret_vec[n_requested] = 1.
        return ret_vec

    def convert_joint_slot_b(self, belief):
        """
        Extracts the features for the joint DIP vector for all the slots
        :param belief: The full belief state
        :return: The DIP joint slot vector
        """
        if type(belief) == DialogueState.DialogueState:
            belief = belief.domainStates[belief.currentdomain]

        joint_beliefs = []
        joint_none = 1.
        informable_beliefs = [copy.deepcopy(belief['beliefs'][x]) for x in belief['beliefs'].keys() if x in self.slots] # this might be inneficent
        for i, b in enumerate(informable_beliefs):
            joint_none *= b['**NONE**']
            del b['**NONE**'] # should I put **NONE** prob mass to dontcare?
            informable_beliefs[i] = sorted([x for x in b.values() if x != 0], reverse=True)[:2]
            while len(informable_beliefs[i]) < 2:
                informable_beliefs[i].append(0.)
        for probs in product(*informable_beliefs):
            joint_beliefs.append(np.prod(probs))
        j_top = joint_beliefs[0]
        j_2nd = joint_beliefs[1]
        j_3rd = joint_beliefs[2]
        first_joint_beliefs = joint_beliefs[:8]
        if sum(first_joint_beliefs) == 0:
            first_joint_beliefs = np.ones(len(first_joint_beliefs)) / len(first_joint_beliefs)
        else:
            first_joint_beliefs = np.array(first_joint_beliefs) / sum(first_joint_beliefs) # why normalise?

        # difference between 1st and 2dn values
        j_ent = entropy(first_joint_beliefs)
        j_dif = joint_beliefs[0] - joint_beliefs[1]
        j_dif_bin = [0.] * 5
        idx = int((j_dif) * 5)
        if idx == 5:
            idx = 4
        j_dif_bin[idx] = 1

        # number of slots which are not **NONE**
        n = 0
        for key in belief['beliefs']:
            if key in self.slots:
                none_val = belief['beliefs'][key]['**NONE**']
                top_val = np.max([belief['beliefs'][key][value] for value in belief['beliefs'][key].keys() if value != '**NONE**'])
                if top_val > none_val:
                    n += 1
        not_none = [0.] * 5
        if n > 4:
            n = 4
        not_none[n] = 1.

        return [j_top, j_2nd, j_3rd, joint_none, j_ent, j_dif] + j_dif_bin + not_none

    def convert_slot_b(self, belief, slot):
        """
        Extracts the slot DIP features.
        :param belief: The full belief state
        :return: The slot DIP vector
        """
        if type(belief) == DialogueState.DialogueState:
            belief = belief.domainStates[belief.currentdomain]
        b = [belief['beliefs'][slot]['**NONE**']] + sorted([belief['beliefs'][slot][value] for value in belief['beliefs'][slot].keys() if value != '**NONE**'], reverse=True)
        b_top = b[1]
        b_2nd = b[2]
        b_3rd = b[3]
        b_ent = entropy(b)
        b_none = b[0]
        b_dif = b[1] - b[2]
        b_dif_bin = [0.] * 5
        idx = int((b_dif) * 5)
        if idx == 5:
            idx = 4
        b_dif_bin[idx] = 1
        non_zero_rate = [x != 0 for x in b[1:]]
        non_zero_rate = sum(non_zero_rate) / len(non_zero_rate)
        requested_prob = belief['beliefs']['requested'][slot]

        # Ontology and DB based features
        V_len = len(Ontology.global_ontology.get_informable_slot_values(self.domainString, slot))
        norm_N_values = 1 / V_len
        v_len_bin_vector = [0.] * self.N_bins
        v_len_bin_vector[int(np.log2(V_len))] = 1.
        #ocurr_prob, not_occur_prob, first_prob, second_prob, later_prob = self._get_importance_and_priority(slot) # this was manually set in the original DIP paper, I think it can be learned from the other features
        val_dist_in_DB = self._get_val_dist_in_DB(slot)
        # potential_contr_to_DB_search = self._get_potential_contr_to_DB_search(slot, belief)
        #potential_contr_to_DB_search = [0, 0, 0, 0] # the implementation of this method is too slow right now, dont knwo how useful these features are (but they seem quite useful)
        return [0, b_top, b_2nd, b_3rd, b_ent, b_none, non_zero_rate, requested_prob, norm_N_values, val_dist_in_DB] + b_dif_bin + v_len_bin_vector

    def _get_val_dist_in_DB(self, slot):
        # The entropy of the normalised histogram (|DB(s=v)|/|DB|) \forall v \in V_s
        values = Ontology.global_ontology.get_informable_slot_values(self.domainString, slot)
        entities = Ontology.global_ontology.entity_by_features(self.domainString, {})
        val_dist = np.zeros(len(values))
        n = 0
        for ent in entities:
            if ent[slot] != 'not available':
                val_dist[values.index(ent[slot])] += 1
                n += 1
        return entropy(val_dist/n)


class padded_state(State):
    def __init__(self, belief, domainString=None, action_freq=None):
        #params
        self.domainString = domainString
        self.sortbelief = False
        #self.action_freq = False
        if Settings.config.has_option('feudalpolicy', 'sortbelief'):
            self.sortbelief = Settings.config.getboolean('feudalpolicy', 'sortbelief')
        #if Settings.config.has_option('feudalpolicy', 'action_freq'):
        #    self.action_freq = Settings.config.getboolean('feudalpolicy', 'action_freq')
        self.slots = list(Ontology.global_ontology.get_informable_slots(domainString))
        if 'price' in self.slots:
            self.slots.remove('price') #remove price from SFR ont, its not used

        if 'name' in self.slots:
            self.slots.remove('name')

        slot_values = Ontology.global_ontology.get_informable_slots_and_values(domainString)
        self.max_v = np.max([len(slot_values[s]) for s in self.slots]) + 3 # (+**NONE**+dontcare+pad)
        self.max_v = 158
        self.si_size = 72 # size of general plus joint vectors
        self.sd_size = self.max_v

        self.DIP_state = {'general':None, 'joint':None}
        for slot in self.slots:
            self.DIP_state[slot]=None

        # convert belief state into DIP params
        if action_freq is not None:
            self.DIP_state['general'] = np.concatenate((action_freq,self.convert_general_b(belief)))
        else:
            self.DIP_state['general'] = self.convert_general_b(belief)
        self.DIP_state['joint'] = self.convert_joint_slot_b(belief)
        for slot in self.slots:
            self.DIP_state[slot] = self.convert_slot_b(belief, slot)

        # create vector and masks
        self.get_DIP_vector()
        self.beliefStateVec = None #for compatibility with GP sarsa implementation

    def get_DIP_vector(self):
        """
        convert the state into a numpy vector and a set of masks per slot
        :return:
        """
        pad_v = np.zeros(len(self.DIP_state[self.slots[0]]))
        slot_len = len(pad_v)
        general_len = len(self.DIP_state['general']) + len(self.DIP_state['joint'])
        pad_v[0] = 1.
        self.DIP_vector = [pad_v]
        self.DIP_masks = {}
        mask_template = [False] * (slot_len * (len(self.slots) + 1)) + [True] * general_len
        i = 1
        for slot in self.slots:
            self.DIP_vector.append(self.DIP_state[slot])
            self.DIP_masks[slot] = np.array(mask_template)
            self.DIP_masks[slot][slot_len*i:slot_len*(i+1)] = True
            i += 1
        self.DIP_vector.append(self.DIP_state['general'])
        self.DIP_vector.append(self.DIP_state['joint'])
        self.DIP_masks['general'] = np.array(mask_template)
        self.DIP_masks['general'][:slot_len] = True

        self.DIP_vector = np.concatenate(self.DIP_vector)

    def get_beliefStateVec(self, slot):
        return self.DIP_vector[self.DIP_masks[slot]]

    def get_DIP_state(self, slot):
        return np.array([self.DIP_state['general'] + self.DIP_state['joint'] + self.DIP_state[slot]])

    def get_full_DIP_state(self):
        full_slot_bstate = []
        for slot in self.slots:
            full_slot_bstate += self.DIP_state[slot]
        full_DIP_state = np.array([full_slot_bstate + self.DIP_state['general'] + self.DIP_state['joint']])
        DIP_mask = [True]*(len(self.DIP_state['general']) + len(self.DIP_state['joint'])) + [False] * len(full_slot_bstate)
        return full_DIP_state, DIP_mask

    def convert_general_b(self, belief):
        """
        Extracts from the belief state the vector corresponding to the general features (e.g. method, user act...)
        :param belief: The full belief state
        :return: The general vector
        """
        if type(belief) == DialogueState.DialogueState:
            belief = belief.domainStates[belief.currentdomain]

        dial_act = belief['beliefs']['discourseAct'].values()

        requested = self._get_requested_vector(belief)
        method = belief['beliefs']['method'].values()
        features = [int(belief['features']['offerHappened']), int(belief['features']['lastActionInformNone']),
                    int(bool(belief['features']['lastInformedVenue']))]
        discriminable = [int(x) for x in belief['features']['inform_info']]

        return dial_act + requested + method + features + discriminable

    def _get_requested_vector(self, belief):
        n_requested = sum([x>0.5 for x in belief['beliefs']['requested'].values()])
        ret_vec = [0] * 5
        if n_requested > 4:
            n_requested = 4
        ret_vec[n_requested] = 1.
        return ret_vec

    def convert_joint_slot_b(self, belief):
        """
            Extracts the features for the joint vector of all the slots
            :param belief: The full belief state
            :return: The joint slot vector
            """
        #ic340 note: this should probably be done with an rnn encoder
        if type(belief) == DialogueState.DialogueState:
            belief = belief.domainStates[belief.currentdomain]

        joint_beliefs = []
        joint_none = 1.
        informable_beliefs = [copy.deepcopy(belief['beliefs'][x]) for x in belief['beliefs'].keys() if
                              x in self.slots]  # this might be inneficent
        for i, b in enumerate(informable_beliefs):
            joint_none *= b['**NONE**']
            del b['**NONE**']  # should I put **NONE** prob mass to dontcare?
            informable_beliefs[i] = sorted([x for x in b.values() if x != 0], reverse=True)[:2]
            while len(informable_beliefs[i]) < 2:
                informable_beliefs[i].append(0.)
        for probs in product(*informable_beliefs):
            joint_beliefs.append(np.prod(probs))
        first_joint_beliefs = -np.ones(20)
        joint_beliefs = joint_beliefs[:20]
        len_joint_beliefs = len(joint_beliefs)
        first_joint_beliefs[:len_joint_beliefs] = joint_beliefs

        if sum(first_joint_beliefs) == 0:
            first_joint_beliefs = list(np.ones(len(first_joint_beliefs)) / len(first_joint_beliefs))
        else:
            first_joint_beliefs = list(np.array(first_joint_beliefs) / sum(first_joint_beliefs))  # why normalise?

        # number of slots which are not **NONE**
        n = 0
        for key in belief['beliefs']:
            if key in self.slots:
                none_val = belief['beliefs'][key]['**NONE**']
                top_val = np.max(
                    [belief['beliefs'][key][value] for value in belief['beliefs'][key].keys() if value != '**NONE**'])
                if top_val > none_val:
                    n += 1
        not_none = [0.] * 5
        if n > 4:
            n = 4
        not_none[n] = 1.

        return [joint_none] + first_joint_beliefs + not_none

    def convert_slot_b(self, belief, slot):
        """
        Extracts the slot features by padding the distribution vector with -1s.
        :param belief: The full belief state
        :return: The slot DIP vector
        """
        if type(belief) == DialogueState.DialogueState:
            belief = belief.domainStates[belief.currentdomain]
        if self.sortbelief is True:
            b = [belief['beliefs'][slot]['**NONE**']] + sorted(
                [belief['beliefs'][slot][value] for value in belief['beliefs'][slot].keys() if value != '**NONE**'],
                reverse=True) # sorted values

        else:
            b = [belief['beliefs'][slot]['**NONE**']] + \
                [belief['beliefs'][slot][value] for value in belief['beliefs'][slot].keys() if value != '**NONE**'] # unsorted values

        assert len(b) <= self.max_v -1, 'length of bstate ({}) is longer than self.max_v ({})'.format(len(b), self.max_v-1)
        padded_b = -np.ones(self.max_v)
        padded_b[0] = 0.
        padded_b[1:len(b)+1] = b
        return padded_b

    def _get_val_dist_in_DB(self, slot):
        # The entropy of the normalised histogram (|DB(s=v)|/|DB|) \forall v \in V_s
        values = Ontology.global_ontology.get_informable_slot_values(self.domainString, slot)
        entities = Ontology.global_ontology.entity_by_features(self.domainString, {})
        val_dist = np.zeros(len(values))
        n = 0
        for ent in entities:
            if ent[slot] != 'not available':
                val_dist[values.index(ent[slot])] += 1
                n += 1
        return entropy(val_dist/n)


def get_test_beliefs():
    b1 = {'beliefs': {u'allowedforkids': {'**NONE**': 0.0,
   u'0': 0.0,
   u'1': 0.0,
   'dontcare': 1.0},
  u'area': {'**NONE**': 1.0,
   u'alamo square': 0.0,
   u'amanico ergina village': 0.0,
   u'anza vista': 0.0,
   u'ashbury heights': 0.0,
   u'balboa terrace': 0.0,
   u'bayview district': 0.0,
   u'bayview heights': 0.0,
   u'bernal heights': 0.0,
   u'bernal heights north': 0.0,
   u'bernal heights south': 0.0,
   u'buena vista park': 0.0,
   u'castro': 0.0,
   u'cathedral hill': 0.0,
   u'cayuga terrace': 0.0,
   u'central richmond': 0.0,
   u'central sunset': 0.0,
   u'central waterfront': 0.0,
   u'chinatown': 0.0,
   u'civic center': 0.0,
   u'clarendon heights': 0.0,
   u'cole valley': 0.0,
   u'corona heights': 0.0,
   u'cow hollow': 0.0,
   u'crocker amazon': 0.0,
   u'diamond heights': 0.0,
   u'doelger city': 0.0,
   u'dogpatch': 0.0,
   u'dolores heights': 0.0,
   'dontcare': 0.0,
   u'downtown': 0.0,
   u'duboce triangle': 0.0,
   u'embarcadero': 0.0,
   u'eureka valley': 0.0,
   u'eureka valley dolores heights': 0.0,
   u'excelsior': 0.0,
   u'financial district': 0.0,
   u'financial district south': 0.0,
   u'fishermans wharf': 0.0,
   u'forest hill': 0.0,
   u'forest hill extension': 0.0,
   u'forest knolls': 0.0,
   u'fort mason': 0.0,
   u'fort winfield scott': 0.0,
   u'frederick douglass haynes gardens': 0.0,
   u'friendship village': 0.0,
   u'glen park': 0.0,
   u'glenridge': 0.0,
   u'golden gate heights': 0.0,
   u'golden gate park': 0.0,
   u'haight ashbury': 0.0,
   u'hayes valley': 0.0,
   u'hunters point': 0.0,
   u'india basin': 0.0,
   u'ingleside': 0.0,
   u'ingleside heights': 0.0,
   u'ingleside terrace': 0.0,
   u'inner mission': 0.0,
   u'inner parkside': 0.0,
   u'inner richmond': 0.0,
   u'inner sunset': 0.0,
   u'inset': 0.0,
   u'jordan park': 0.0,
   u'laguna honda': 0.0,
   u'lake': 0.0,
   u'lake shore': 0.0,
   u'lakeside': 0.0,
   u'laurel heights': 0.0,
   u'lincoln park': 0.0,
   u'lincoln park lobos': 0.0,
   u'little hollywood': 0.0,
   u'little italy': 0.0,
   u'little osaka': 0.0,
   u'little russia': 0.0,
   u'lone mountain': 0.0,
   u'lower haight': 0.0,
   u'lower nob hill': 0.0,
   u'lower pacific heights': 0.0,
   u'malcolm x square': 0.0,
   u'marcus garvey square': 0.0,
   u'marina district': 0.0,
   u'martin luther king square': 0.0,
   u'mastro': 0.0,
   u'merced heights': 0.0,
   u'merced manor': 0.0,
   u'midtown terrace': 0.0,
   u'miraloma park': 0.0,
   u'mission bay': 0.0,
   u'mission district': 0.0,
   u'mission dolores': 0.0,
   u'mission terrace': 0.0,
   u'monterey heights': 0.0,
   u'mount davidson manor': 0.0,
   u'nob hill': 0.0,
   u'noe valley': 0.0,
   u'noma': 0.0,
   u'north beach': 0.0,
   u'north panhandle': 0.0,
   u'north park': 0.0,
   u'north waterfront': 0.0,
   u'oceanview': 0.0,
   u'opera plaza': 0.0,
   u'outer mission': 0.0,
   u'outer parkside': 0.0,
   u'outer richmond': 0.0,
   u'outer sunset': 0.0,
   u'outset': 0.0,
   u'pacific heights': 0.0,
   u'panhandle': 0.0,
   u'park merced': 0.0,
   u'parkmerced': 0.0,
   u'parkside': 0.0,
   u'pine lake park': 0.0,
   u'portola': 0.0,
   u'potrero flats': 0.0,
   u'potrero hill': 0.0,
   u'presidio': 0.0,
   u'presidio heights': 0.0,
   u'richmond district': 0.0,
   u'russian hill': 0.0,
   u'saint francis wood': 0.0,
   u'san francisco airport': 0.0,
   u'san francisco state university': 0.0,
   u'sea cliff': 0.0,
   u'sherwood forest': 0.0,
   u'showplace square': 0.0,
   u'silver terrace': 0.0,
   u'somisspo': 0.0,
   u'south basin': 0.0,
   u'south beach': 0.0,
   u'south of market': 0.0,
   u'st francis square': 0.0,
   u'st francis wood': 0.0,
   u'stonestown': 0.0,
   u'sunnydale': 0.0,
   u'sunnyside': 0.0,
   u'sunset district': 0.0,
   u'telegraph hill': 0.0,
   u'tenderloin': 0.0,
   u'thomas paine square': 0.0,
   u'transmission': 0.0,
   u'treasure island': 0.0,
   u'twin peaks': 0.0,
   u'twin peaks west': 0.0,
   u'upper market': 0.0,
   u'van ness': 0.0,
   u'victoria mews': 0.0,
   u'visitacion valley': 0.0,
   u'vista del monte': 0.0,
   u'west of twin peaks': 0.0,
   u'west portal': 0.0,
   u'western addition': 0.0,
   u'westlake and olympic': 0.0,
   u'westwood highlands': 0.0,
   u'westwood park': 0.0,
   u'yerba buena island': 0.0,
   u'zion district': 0.0},
  'discourseAct': {u'ack': 0.0,
   'bye': 0.0,
   u'hello': 0.0,
   u'none': 1.0,
   u'repeat': 0.0,
   u'silence': 0.0,
   u'thankyou': 0.0},
  u'food': {'**NONE**': 0.0,
   u'afghan': 0.0,
   u'arabian': 0.0,
   u'asian': 0.0,
   u'basque': 0.0,
   u'brasseries': 0.0,
   u'brazilian': 0.0,
   u'buffets': 0.0,
   u'burgers': 0.0,
   u'burmese': 0.0,
   u'cafes': 0.0,
   u'cambodian': 0.0,
   u'cantonese': 1.0,
   u'chinese': 0.0,
   u'comfort food': 0.0,
   u'creperies': 0.0,
   u'dim sum': 0.0,
   'dontcare': 0.0,
   u'ethiopian': 0.0,
   u'ethnic food': 0.0,
   u'french': 0.0,
   u'gluten free': 0.0,
   u'himalayan': 0.0,
   u'indian': 0.0,
   u'indonesian': 0.0,
   u'indpak': 0.0,
   u'italian': 0.0,
   u'japanese': 0.0,
   u'korean': 0.0,
   u'kosher': 0.0,
   u'latin': 0.0,
   u'lebanese': 0.0,
   u'lounges': 0.0,
   u'malaysian': 0.0,
   u'mediterranean': 0.0,
   u'mexican': 0.0,
   u'middle eastern': 0.0,
   u'modern european': 0.0,
   u'moroccan': 0.0,
   u'new american': 0.0,
   u'pakistani': 0.0,
   u'persian': 0.0,
   u'peruvian': 0.0,
   u'pizza': 0.0,
   u'raw food': 0.0,
   u'russian': 0.0,
   u'sandwiches': 0.0,
   u'sea food': 0.0,
   u'shanghainese': 0.0,
   u'singaporean': 0.0,
   u'soul food': 0.0,
   u'spanish': 0.0,
   u'steak': 0.0,
   u'sushi': 0.0,
   u'taiwanese': 0.0,
   u'tapas': 0.0,
   u'thai': 0.0,
   u'traditionnal american': 0.0,
   u'turkish': 0.0,
   u'vegetarian': 0.0,
   u'vietnamese': 0.0},
  u'goodformeal': {'**NONE**': 0.0,
   u'breakfast': 0.0,
   u'brunch': 0.0,
   u'dinner': 0.0,
   'dontcare': 1.0,
   u'lunch': 0.0},
  'method': {u'byalternatives': 0.0,
   u'byconstraints': 0.0,
   u'byname': 0.9285714285714286,
   u'finished': 0.0,
   u'none': 0.0714285714285714,
   u'restart': 0.0},
  u'name': {'**NONE**': 0.0,
   u'a 16': 0.0,
   u'a la turca restaurant': 0.0,
   u'abacus': 0.0,
   u'alamo square seafood grill': 0.0,
   u'albona ristorante istriano': 0.0,
   u'alborz persian cuisine': 0.0,
   u'allegro romano': 0.0,
   u'amarena': 0.0,
   u'amber india': 0.0,
   u'ame': 0.0,
   u'ananda fuara': 0.0,
   u'anchor oyster bar': 0.0,
   u'angkor borei restaurant': 0.0,
   u'aperto restaurant': 0.0,
   u'ar roi restaurant': 0.0,
   u'arabian nights restaurant': 0.0,
   u'assab eritrean restaurant': 0.0,
   u'atelier crenn': 0.0,
   u'aux delices restaurant': 0.0,
   u'aziza': 0.0,
   u'b star bar': 0.0,
   u'bar crudo': 0.0,
   u'beijing restaurant': 0.0,
   u'bella trattoria': 0.0,
   u'benu': 0.0,
   u'betelnut': 0.0,
   u'bistro central parc': 0.0,
   u'bix': 0.0,
   u'borgo': 0.0,
   u'borobudur restaurant': 0.0,
   u'bouche': 0.0,
   u'boulevard': 0.0,
   u'brothers restaurant': 0.0,
   u'bund shanghai restaurant': 0.0,
   u'burma superstar': 0.0,
   u'butterfly': 0.0,
   u'cafe claude': 0.0,
   u'cafe jacqueline': 0.0,
   u'campton place restaurant': 0.0,
   u'canteen': 0.0,
   u'canto do brasil restaurant': 0.0,
   u'capannina': 0.0,
   u'capital restaurant': 0.0,
   u'chai yo thai restaurant': 0.0,
   u'chaya brasserie': 0.0,
   u'chenery park': 0.0,
   u'chez maman': 0.0,
   u'chez papa bistrot': 0.0,
   u'chez spencer': 0.0,
   u'chiaroscuro': 0.0,
   u'chouchou': 0.0,
   u'chow': 0.0,
   u'city view restaurant': 0.0,
   u'claudine': 0.0,
   u'coi': 0.0,
   u'colibri mexican bistro': 0.0,
   u'coqueta': 0.0,
   u'crustacean restaurant': 0.0,
   u'da flora a venetian osteria': 0.0,
   u'darbar restaurant': 0.0,
   u'delancey street restaurant': 0.0,
   u'delfina': 0.0,
   u'dong baek restaurant': 0.0,
   'dontcare': 0.0,
   u'dosa on fillmore': 0.0,
   u'dosa on valencia': 0.0,
   u'eiji': 0.0,
   u'enjoy vegetarian restaurant': 0.0,
   u'espetus churrascaria': 0.0,
   u'fang': 0.0,
   u'farallon': 0.0,
   u'fattoush restaurant': 0.0,
   u'fifth floor': 0.0,
   u'fino restaurant': 0.0,
   u'firefly': 0.0,
   u'firenze by night ristorante': 0.0,
   u'fleur de lys': 0.0,
   u'fog harbor fish house': 0.0,
   u'forbes island': 0.0,
   u'foreign cinema': 0.0,
   u'frances': 0.0,
   u'franchino': 0.0,
   u'franciscan crab restaurant': 0.0,
   u'frascati': 0.0,
   u'fresca': 0.0,
   u'fringale': 0.0,
   u'fujiyama ya japanese restaurant': 0.0,
   u'gajalee': 0.0,
   u'gamine': 0.0,
   u'garcon restaurant': 0.0,
   u'gary danko': 0.0,
   u'gitane': 0.0,
   u'golden era restaurant': 0.0,
   u'gracias madre': 0.0,
   u'great eastern restaurant': 1.0,
   u'hakka restaurant': 0.0,
   u'hakkasan': 0.0,
   u'han second kwan': 0.0,
   u'heirloom cafe': 0.0,
   u'helmand palace': 0.0,
   u'hi dive': 0.0,
   u'hillside supper club': 0.0,
   u'hillstone': 0.0,
   u'hong kong clay pot restaurant': 0.0,
   u'house of nanking': 0.0,
   u'house of prime rib': 0.0,
   u'hunan homes restaurant': 0.0,
   u'incanto': 0.0,
   u'isa': 0.0,
   u'jannah': 0.0,
   u'jasmine garden': 0.0,
   u'jitlada thai cuisine': 0.0,
   u'kappa japanese restaurant': 0.0,
   u'kim thanh restaurant': 0.0,
   u'kirin chinese restaurant': 0.0,
   u'kiss seafood': 0.0,
   u'kokkari estiatorio': 0.0,
   u'la briciola': 0.0,
   u'la ciccia': 0.0,
   u'la folie': 0.0,
   u'la mediterranee': 0.0,
   u'la traviata': 0.0,
   u'lahore karahi': 0.0,
   u'lavash': 0.0,
   u'le charm': 0.0,
   u'le colonial': 0.0,
   u'le soleil': 0.0,
   u'lime tree southeast asian kitchen': 0.0,
   u'little delhi': 0.0,
   u'little nepal': 0.0,
   u'luce': 0.0,
   u'lucky creation restaurant': 0.0,
   u'luella': 0.0,
   u'lupa': 0.0,
   u'm y china': 0.0,
   u'maki restaurant': 0.0,
   u'mangia tutti ristorante': 0.0,
   u'manna': 0.0,
   u'marlowe': 0.0,
   u'marnee thai': 0.0,
   u'maverick': 0.0,
   u'mela tandoori kitchen': 0.0,
   u'mescolanza': 0.0,
   u'mezes': 0.0,
   u'michael mina restaurant': 0.0,
   u'millennium': 0.0,
   u'minako organic japanese restaurant': 0.0,
   u'minami restaurant': 0.0,
   u'mission chinese food': 0.0,
   u'mochica': 0.0,
   u'modern thai': 0.0,
   u'mona lisa restaurant': 0.0,
   u'mozzeria': 0.0,
   u'muguboka restaurant': 0.0,
   u'my tofu house': 0.0,
   u'nicaragua restaurant': 0.0,
   u'nob hill cafe': 0.0,
   u'nopa': 0.0,
   u'old jerusalem restaurant': 0.0,
   u'old skool cafe': 0.0,
   u'one market restaurant': 0.0,
   u'orexi': 0.0,
   u'original us restaurant': 0.0,
   u'osha thai': 0.0,
   u'oyaji restaurant': 0.0,
   u'ozumo': 0.0,
   u'pad thai restaurant': 0.0,
   u'panta rei restaurant': 0.0,
   u'park tavern': 0.0,
   u'pera': 0.0,
   u'piperade': 0.0,
   u'ploy 2': 0.0,
   u'poc chuc': 0.0,
   u'poesia': 0.0,
   u'prospect': 0.0,
   u'quince': 0.0,
   u'radius san francisco': 0.0,
   u'range': 0.0,
   u'red door cafe': 0.0,
   u'restaurant ducroix': 0.0,
   u'ristorante bacco': 0.0,
   u'ristorante ideale': 0.0,
   u'ristorante milano': 0.0,
   u'ristorante parma': 0.0,
   u'rn74': 0.0,
   u'rue lepic': 0.0,
   u'saha': 0.0,
   u'sai jai thai restaurant': 0.0,
   u'salt house': 0.0,
   u'san tung chinese restaurant': 0.0,
   u'san wang restaurant': 0.0,
   u'sanjalisco': 0.0,
   u'sanraku': 0.0,
   u'seasons': 0.0,
   u'seoul garden': 0.0,
   u'seven hills': 0.0,
   u'shangri la vegetarian restaurant': 0.0,
   u'singapore malaysian restaurant': 0.0,
   u'skool': 0.0,
   u'so': 0.0,
   u'sotto mare': 0.0,
   u'source': 0.0,
   u'specchio ristorante': 0.0,
   u'spruce': 0.0,
   u'straits restaurant': 0.0,
   u'stroganoff restaurant': 0.0,
   u'sunflower potrero hill': 0.0,
   u'sushi bistro': 0.0,
   u'taiwan restaurant': 0.0,
   u'tanuki restaurant': 0.0,
   u'tataki': 0.0,
   u'tekka japanese restaurant': 0.0,
   u'thai cottage restaurant': 0.0,
   u'thai house express': 0.0,
   u'thai idea vegetarian': 0.0,
   u'thai time restaurant': 0.0,
   u'thanh long': 0.0,
   u'the big 4 restaurant': 0.0,
   u'the blue plate': 0.0,
   u'the house': 0.0,
   u'the richmond': 0.0,
   u'the slanted door': 0.0,
   u'the stinking rose': 0.0,
   u'thep phanom thai restaurant': 0.0,
   u'tommys joynt': 0.0,
   u'toraya japanese restaurant': 0.0,
   u'town hall': 0.0,
   u'trattoria contadina': 0.0,
   u'tu lan': 0.0,
   u'tuba restaurant': 0.0,
   u'u lee restaurant': 0.0,
   u'udupi palace': 0.0,
   u'venticello ristorante': 0.0,
   u'vicoletto': 0.0,
   u'yank sing': 0.0,
   u'yummy yummy': 0.0,
   u'z and y restaurant': 0.0,
   u'zadin': 0.0,
   u'zare at fly trap': 0.0,
   u'zarzuela': 0.0,
   u'zen yai thai restaurant': 0.0,
   u'zuni cafe': 0.0,
   u'zushi puzzle': 0.0},
  u'near': {'**NONE**': 0.0,
   u'bayview hunters point': 0.0,
   'dontcare': 1.0,
   u'haight': 0.0,
   u'japantown': 0.0,
   u'marina cow hollow': 0.0,
   u'mission': 0.0,
   u'nopa': 0.0,
   u'north beach telegraph hill': 0.0,
   u'soma': 0.0,
   u'union square': 0.0},
  u'price': {'**NONE**': 1.0,
   u'10 dollar': 0.0,
   u'10 euro': 0.0,
   u'11 euro': 0.0,
   u'15 euro': 0.0,
   u'18 euro': 0.0,
   u'20 euro': 0.0,
   u'22 euro': 0.0,
   u'25 euro': 0.0,
   u'26 euro': 0.0,
   u'29 euro': 0.0,
   u'37 euro': 0.0,
   u'6': 0.0,
   u'7': 0.0,
   u'9': 0.0,
   u'between 0 and 15 euro': 0.0,
   u'between 10 and 13 euro': 0.0,
   u'between 10 and 15 euro': 0.0,
   u'between 10 and 18 euro': 0.0,
   u'between 10 and 20 euro': 0.0,
   u'between 10 and 23 euro': 0.0,
   u'between 10 and 30 euro': 0.0,
   u'between 11 and 15 euro': 0.0,
   u'between 11 and 18 euro': 0.0,
   u'between 11 and 22 euro': 0.0,
   u'between 11 and 25 euro': 0.0,
   u'between 11 and 29 euro': 0.0,
   u'between 11 and 35 euro': 0.0,
   u'between 13 and 15 euro': 0.0,
   u'between 13 and 18 euro': 0.0,
   u'between 13 and 24 euro': 0.0,
   u'between 15 and 18 euro': 0.0,
   u'between 15 and 22 euro': 0.0,
   u'between 15 and 26 euro': 0.0,
   u'between 15 and 29 euro': 0.0,
   u'between 15 and 33 euro': 0.0,
   u'between 15 and 44 euro': 0.0,
   u'between 15 and 58 euro': 0.0,
   u'between 18 and 26 euro': 0.0,
   u'between 18 and 29 euro': 0.0,
   u'between 18 and 44 euro': 0.0,
   u'between 18 and 55 euro': 0.0,
   u'between 18 and 58 euro': 0.0,
   u'between 18 and 73 euro': 0.0,
   u'between 18 and 78 euro': 0.0,
   u'between 2 and 15 euro': 0.0,
   u'between 20 and 30 euro': 0.0,
   u'between 21 and 23 euro': 0.0,
   u'between 22 and 29 euro': 0.0,
   u'between 22 and 30 dollar': 0.0,
   u'between 22 and 37 euro': 0.0,
   u'between 22 and 58 euro': 0.0,
   u'between 22 and 73 euro': 0.0,
   u'between 23 and 29': 0.0,
   u'between 23 and 29 euro': 0.0,
   u'between 23 and 37 euro': 0.0,
   u'between 23 and 58': 0.0,
   u'between 23 and 58 euro': 0.0,
   u'between 26 and 33 euro': 0.0,
   u'between 26 and 34 euro': 0.0,
   u'between 26 and 37 euro': 0.0,
   u'between 29 and 37 euro': 0.0,
   u'between 29 and 44 euro': 0.0,
   u'between 29 and 58 euro': 0.0,
   u'between 29 and 73 euro': 0.0,
   u'between 30 and 58': 0.0,
   u'between 30 and 58 euro': 0.0,
   u'between 31 and 50 euro': 0.0,
   u'between 37 and 110 euro': 0.0,
   u'between 37 and 44 euro': 0.0,
   u'between 37 and 58 euro': 0.0,
   u'between 4 and 22 euro': 0.0,
   u'between 4 and 58 euro': 0.0,
   u'between 5 an 30 euro': 0.0,
   u'between 5 and 10 euro': 0.0,
   u'between 5 and 11 euro': 0.0,
   u'between 5 and 15 dollar': 0.0,
   u'between 5 and 20 euro': 0.0,
   u'between 5 and 25 euro': 0.0,
   u'between 6 and 10 euro': 0.0,
   u'between 6 and 11 euro': 0.0,
   u'between 6 and 15 euro': 0.0,
   u'between 6 and 29 euro': 0.0,
   u'between 7 and 11 euro': 0.0,
   u'between 7 and 13 euro': 0.0,
   u'between 7 and 15 euro': 0.0,
   u'between 7 and 37 euro': 0.0,
   u'between 8 and 22 euro': 0.0,
   u'between 9 and 13 dolllar': 0.0,
   u'between 9 and 15 euro': 0.0,
   u'between 9 and 58 euro': 0.0,
   u'bteween 11 and 15 euro': 0.0,
   u'bteween 15 and 22 euro': 0.0,
   u'bteween 22 and 37': 0.0,
   u'bteween 30 and 58 euro': 0.0,
   u'bteween 51 and 73 euro': 0.0,
   u'netween 20 and 30 euro': 0.0},
  u'pricerange': {'**NONE**': 1.0,
   u'cheap': 0.0,
   'dontcare': 0.0,
   u'expensive': 0.0,
   u'moderate': 0.0},
  'requested': {u'addr': 1.0,
   u'allowedforkids': 0.0,
   u'area': 0.0,
   u'food': 0.0,
   u'goodformeal': 0.0,
   u'name': 0.0,
   u'near': 0.0,
   u'phone': 1,
   u'postcode': 0.0,
   u'price': 0.0,
   u'pricerange': 0.0}},
 'features': {'inform_info': [False,
   False,
   True,
   False,
   True,
   False,
   False,
   True,
   False,
   True,
   False,
   False,
   True,
   False,
   True,
   False,
   False,
   True,
   False,
   True,
   False,
   False,
   True,
   False,
   True],
  'informedVenueSinceNone': ['great eastern restaurant',
   'great eastern restaurant'],
  'lastActionInformNone': False,
  'lastInformedVenue': 'great eastern restaurant',
  'offerHappened': False},
 'userActs': [(u'request(name="great eastern restaurant",phone)', 1.0)]}
    b2 = {'beliefs': {u'allowedforkids': {'**NONE**': 0.014367834316388661,
   u'0': 0.009175995595522114,
   u'1': 0.9579333306577846,
   'dontcare': 0.01852283943030468},
  u'area': {'**NONE**': 0.9753165718480455,
   u'alamo square': 0.0,
   u'amanico ergina village': 0.0,
   u'anza vista': 0.0,
   u'ashbury heights': 0.0,
   u'balboa terrace': 0.0,
   u'bayview district': 0.0,
   u'bayview heights': 0.0,
   u'bernal heights': 0.0,
   u'bernal heights north': 0.0,
   u'bernal heights south': 0.0,
   u'buena vista park': 0.0,
   u'castro': 0.0,
   u'cathedral hill': 0.0,
   u'cayuga terrace': 0.0,
   u'central richmond': 0.0,
   u'central sunset': 0.0,
   u'central waterfront': 0.0,
   u'chinatown': 0.0,
   u'civic center': 0.0,
   u'clarendon heights': 0.0,
   u'cole valley': 0.0,
   u'corona heights': 0.0,
   u'cow hollow': 0.0,
   u'crocker amazon': 0.0,
   u'diamond heights': 0.0,
   u'doelger city': 0.0,
   u'dogpatch': 0.0,
   u'dolores heights': 0.0,
   'dontcare': 0.0,
   u'downtown': 0.0,
   u'duboce triangle': 0.0,
   u'embarcadero': 0.0,
   u'eureka valley': 0.0,
   u'eureka valley dolores heights': 0.0,
   u'excelsior': 0.0,
   u'financial district': 0.0,
   u'financial district south': 0.0,
   u'fishermans wharf': 0.0,
   u'forest hill': 0.0,
   u'forest hill extension': 0.0,
   u'forest knolls': 0.0,
   u'fort mason': 0.0,
   u'fort winfield scott': 0.0,
   u'frederick douglass haynes gardens': 0.0,
   u'friendship village': 0.0,
   u'glen park': 0.0,
   u'glenridge': 0.0,
   u'golden gate heights': 0.0,
   u'golden gate park': 0.0,
   u'haight ashbury': 0.0,
   u'hayes valley': 0.0,
   u'hunters point': 0.0,
   u'india basin': 0.0,
   u'ingleside': 0.0,
   u'ingleside heights': 0.0,
   u'ingleside terrace': 0.0,
   u'inner mission': 0.0,
   u'inner parkside': 0.0,
   u'inner richmond': 0.0,
   u'inner sunset': 0.0,
   u'inset': 0.0,
   u'jordan park': 0.0,
   u'laguna honda': 0.0,
   u'lake': 0.0,
   u'lake shore': 0.0,
   u'lakeside': 0.0,
   u'laurel heights': 0.0,
   u'lincoln park': 0.0,
   u'lincoln park lobos': 0.0,
   u'little hollywood': 0.0,
   u'little italy': 0.0,
   u'little osaka': 0.0,
   u'little russia': 0.0,
   u'lone mountain': 0.0,
   u'lower haight': 0.0,
   u'lower nob hill': 0.0,
   u'lower pacific heights': 0.0,
   u'malcolm x square': 0.0,
   u'marcus garvey square': 0.0,
   u'marina district': 0.0,
   u'martin luther king square': 0.0,
   u'mastro': 0.0,
   u'merced heights': 0.0,
   u'merced manor': 0.0,
   u'midtown terrace': 0.0,
   u'miraloma park': 0.0,
   u'mission bay': 0.0,
   u'mission district': 0.0,
   u'mission dolores': 0.0,
   u'mission terrace': 0.0,
   u'monterey heights': 0.0,
   u'mount davidson manor': 0.0,
   u'nob hill': 0.0,
   u'noe valley': 0.0,
   u'noma': 0.0,
   u'north beach': 0.0,
   u'north panhandle': 0.0,
   u'north park': 0.0,
   u'north waterfront': 0.0,
   u'oceanview': 0.0,
   u'opera plaza': 0.0,
   u'outer mission': 0.0,
   u'outer parkside': 0.0,
   u'outer richmond': 0.0,
   u'outer sunset': 0.0,
   u'outset': 0.0,
   u'pacific heights': 0.0,
   u'panhandle': 0.0,
   u'park merced': 0.0,
   u'parkmerced': 0.0,
   u'parkside': 0.0,
   u'pine lake park': 0.0,
   u'portola': 0.0,
   u'potrero flats': 0.0,
   u'potrero hill': 0.0,
   u'presidio': 0.0,
   u'presidio heights': 0.0,
   u'richmond district': 0.0,
   u'russian hill': 0.0,
   u'saint francis wood': 0.0,
   u'san francisco airport': 0.0,
   u'san francisco state university': 0.0,
   u'sea cliff': 0.0,
   u'sherwood forest': 0.0,
   u'showplace square': 0.0,
   u'silver terrace': 0.0,
   u'somisspo': 0.0,
   u'south basin': 0.0,
   u'south beach': 0.0,
   u'south of market': 0.0,
   u'st francis square': 0.0,
   u'st francis wood': 0.0,
   u'stonestown': 0.024683428151954484,
   u'sunnydale': 0.0,
   u'sunnyside': 0.0,
   u'sunset district': 0.0,
   u'telegraph hill': 0.0,
   u'tenderloin': 0.0,
   u'thomas paine square': 0.0,
   u'transmission': 0.0,
   u'treasure island': 0.0,
   u'twin peaks': 0.0,
   u'twin peaks west': 0.0,
   u'upper market': 0.0,
   u'van ness': 0.0,
   u'victoria mews': 0.0,
   u'visitacion valley': 0.0,
   u'vista del monte': 0.0,
   u'west of twin peaks': 0.0,
   u'west portal': 0.0,
   u'western addition': 0.0,
   u'westlake and olympic': 0.0,
   u'westwood highlands': 0.0,
   u'westwood park': 0.0,
   u'yerba buena island': 0.0,
   u'zion district': 0.0},
  'discourseAct': {u'ack': 0.0,
   'bye': 0.0,
   u'hello': 0.0,
   u'none': 0.9999999999999998,
   u'repeat': 0.0,
   u'silence': 0.0,
   u'thankyou': 0.0},
  u'food': {'**NONE**': 1.0,
   u'afghan': 0.0,
   u'arabian': 0.0,
   u'asian': 0.0,
   u'basque': 0.0,
   u'brasseries': 0.0,
   u'brazilian': 0.0,
   u'buffets': 0.0,
   u'burgers': 0.0,
   u'burmese': 0.0,
   u'cafes': 0.0,
   u'cambodian': 0.0,
   u'cantonese': 0.0,
   u'chinese': 0.0,
   u'comfort food': 0.0,
   u'creperies': 0.0,
   u'dim sum': 0.0,
   'dontcare': 0.0,
   u'ethiopian': 0.0,
   u'ethnic food': 0.0,
   u'french': 0.0,
   u'gluten free': 0.0,
   u'himalayan': 0.0,
   u'indian': 0.0,
   u'indonesian': 0.0,
   u'indpak': 0.0,
   u'italian': 0.0,
   u'japanese': 0.0,
   u'korean': 0.0,
   u'kosher': 0.0,
   u'latin': 0.0,
   u'lebanese': 0.0,
   u'lounges': 0.0,
   u'malaysian': 0.0,
   u'mediterranean': 0.0,
   u'mexican': 0.0,
   u'middle eastern': 0.0,
   u'modern european': 0.0,
   u'moroccan': 0.0,
   u'new american': 0.0,
   u'pakistani': 0.0,
   u'persian': 0.0,
   u'peruvian': 0.0,
   u'pizza': 0.0,
   u'raw food': 0.0,
   u'russian': 0.0,
   u'sandwiches': 0.0,
   u'sea food': 0.0,
   u'shanghainese': 0.0,
   u'singaporean': 0.0,
   u'soul food': 0.0,
   u'spanish': 0.0,
   u'steak': 0.0,
   u'sushi': 0.0,
   u'taiwanese': 0.0,
   u'tapas': 0.0,
   u'thai': 0.0,
   u'traditionnal american': 0.0,
   u'turkish': 0.0,
   u'vegetarian': 0.0,
   u'vietnamese': 0.0},
  u'goodformeal': {'**NONE**': 1.0,
   u'breakfast': 0.0,
   u'brunch': 0.0,
   u'dinner': 0.0,
   'dontcare': 0.0,
   u'lunch': 0.0},
  'method': {u'byalternatives': 0.0,
   u'byconstraints': 0.7725475751076113,
   u'byname': 0.0,
   u'finished': 0.0,
   u'none': 0.0,
   u'restart': 0.0},
  u'name': {'**NONE**': 1.0,
   u'a 16': 0.0,
   u'a la turca restaurant': 0.0,
   u'abacus': 0.0,
   u'alamo square seafood grill': 0.0,
   u'albona ristorante istriano': 0.0,
   u'alborz persian cuisine': 0.0,
   u'allegro romano': 0.0,
   u'amarena': 0.0,
   u'amber india': 0.0,
   u'ame': 0.0,
   u'ananda fuara': 0.0,
   u'anchor oyster bar': 0.0,
   u'angkor borei restaurant': 0.0,
   u'aperto restaurant': 0.0,
   u'ar roi restaurant': 0.0,
   u'arabian nights restaurant': 0.0,
   u'assab eritrean restaurant': 0.0,
   u'atelier crenn': 0.0,
   u'aux delices restaurant': 0.0,
   u'aziza': 0.0,
   u'b star bar': 0.0,
   u'bar crudo': 0.0,
   u'beijing restaurant': 0.0,
   u'bella trattoria': 0.0,
   u'benu': 0.0,
   u'betelnut': 0.0,
   u'bistro central parc': 0.0,
   u'bix': 0.0,
   u'borgo': 0.0,
   u'borobudur restaurant': 0.0,
   u'bouche': 0.0,
   u'boulevard': 0.0,
   u'brothers restaurant': 0.0,
   u'bund shanghai restaurant': 0.0,
   u'burma superstar': 0.0,
   u'butterfly': 0.0,
   u'cafe claude': 0.0,
   u'cafe jacqueline': 0.0,
   u'campton place restaurant': 0.0,
   u'canteen': 0.0,
   u'canto do brasil restaurant': 0.0,
   u'capannina': 0.0,
   u'capital restaurant': 0.0,
   u'chai yo thai restaurant': 0.0,
   u'chaya brasserie': 0.0,
   u'chenery park': 0.0,
   u'chez maman': 0.0,
   u'chez papa bistrot': 0.0,
   u'chez spencer': 0.0,
   u'chiaroscuro': 0.0,
   u'chouchou': 0.0,
   u'chow': 0.0,
   u'city view restaurant': 0.0,
   u'claudine': 0.0,
   u'coi': 0.0,
   u'colibri mexican bistro': 0.0,
   u'coqueta': 0.0,
   u'crustacean restaurant': 0.0,
   u'da flora a venetian osteria': 0.0,
   u'darbar restaurant': 0.0,
   u'delancey street restaurant': 0.0,
   u'delfina': 0.0,
   u'dong baek restaurant': 0.0,
   u'dosa on fillmore': 0.0,
   u'dosa on valencia': 0.0,
   u'eiji': 0.0,
   u'enjoy vegetarian restaurant': 0.0,
   u'espetus churrascaria': 0.0,
   u'fang': 0.0,
   u'farallon': 0.0,
   u'fattoush restaurant': 0.0,
   u'fifth floor': 0.0,
   u'fino restaurant': 0.0,
   u'firefly': 0.0,
   u'firenze by night ristorante': 0.0,
   u'fleur de lys': 0.0,
   u'fog harbor fish house': 0.0,
   u'forbes island': 0.0,
   u'foreign cinema': 0.0,
   u'frances': 0.0,
   u'franchino': 0.0,
   u'franciscan crab restaurant': 0.0,
   u'frascati': 0.0,
   u'fresca': 0.0,
   u'fringale': 0.0,
   u'fujiyama ya japanese restaurant': 0.0,
   u'gajalee': 0.0,
   u'gamine': 0.0,
   u'garcon restaurant': 0.0,
   u'gary danko': 0.0,
   u'gitane': 0.0,
   u'golden era restaurant': 0.0,
   u'gracias madre': 0.0,
   u'great eastern restaurant': 0.0,
   u'hakka restaurant': 0.0,
   u'hakkasan': 0.0,
   u'han second kwan': 0.0,
   u'heirloom cafe': 0.0,
   u'helmand palace': 0.0,
   u'hi dive': 0.0,
   u'hillside supper club': 0.0,
   u'hillstone': 0.0,
   u'hong kong clay pot restaurant': 0.0,
   u'house of nanking': 0.0,
   u'house of prime rib': 0.0,
   u'hunan homes restaurant': 0.0,
   u'incanto': 0.0,
   u'isa': 0.0,
   u'jannah': 0.0,
   u'jasmine garden': 0.0,
   u'jitlada thai cuisine': 0.0,
   u'kappa japanese restaurant': 0.0,
   u'kim thanh restaurant': 0.0,
   u'kirin chinese restaurant': 0.0,
   u'kiss seafood': 0.0,
   u'kokkari estiatorio': 0.0,
   u'la briciola': 0.0,
   u'la ciccia': 0.0,
   u'la folie': 0.0,
   u'la mediterranee': 0.0,
   u'la traviata': 0.0,
   u'lahore karahi': 0.0,
   u'lavash': 0.0,
   u'le charm': 0.0,
   u'le colonial': 0.0,
   u'le soleil': 0.0,
   u'lime tree southeast asian kitchen': 0.0,
   u'little delhi': 0.0,
   u'little nepal': 0.0,
   u'luce': 0.0,
   u'lucky creation restaurant': 0.0,
   u'luella': 0.0,
   u'lupa': 0.0,
   u'm y china': 0.0,
   u'maki restaurant': 0.0,
   u'mangia tutti ristorante': 0.0,
   u'manna': 0.0,
   u'marlowe': 0.0,
   u'marnee thai': 0.0,
   u'maverick': 0.0,
   u'mela tandoori kitchen': 0.0,
   u'mescolanza': 0.0,
   u'mezes': 0.0,
   u'michael mina restaurant': 0.0,
   u'millennium': 0.0,
   u'minako organic japanese restaurant': 0.0,
   u'minami restaurant': 0.0,
   u'mission chinese food': 0.0,
   u'mochica': 0.0,
   u'modern thai': 0.0,
   u'mona lisa restaurant': 0.0,
   u'mozzeria': 0.0,
   u'muguboka restaurant': 0.0,
   u'my tofu house': 0.0,
   u'nicaragua restaurant': 0.0,
   u'nob hill cafe': 0.0,
   u'nopa': 0.0,
   u'old jerusalem restaurant': 0.0,
   u'old skool cafe': 0.0,
   u'one market restaurant': 0.0,
   u'orexi': 0.0,
   u'original us restaurant': 0.0,
   u'osha thai': 0.0,
   u'oyaji restaurant': 0.0,
   u'ozumo': 0.0,
   u'pad thai restaurant': 0.0,
   u'panta rei restaurant': 0.0,
   u'park tavern': 0.0,
   u'pera': 0.0,
   u'piperade': 0.0,
   u'ploy 2': 0.0,
   u'poc chuc': 0.0,
   u'poesia': 0.0,
   u'prospect': 0.0,
   u'quince': 0.0,
   u'radius san francisco': 0.0,
   u'range': 0.0,
   u'red door cafe': 0.0,
   u'restaurant ducroix': 0.0,
   u'ristorante bacco': 0.0,
   u'ristorante ideale': 0.0,
   u'ristorante milano': 0.0,
   u'ristorante parma': 0.0,
   u'rn74': 0.0,
   u'rue lepic': 0.0,
   u'saha': 0.0,
   u'sai jai thai restaurant': 0.0,
   u'salt house': 0.0,
   u'san tung chinese restaurant': 0.0,
   u'san wang restaurant': 0.0,
   u'sanjalisco': 0.0,
   u'sanraku': 0.0,
   u'seasons': 0.0,
   u'seoul garden': 0.0,
   u'seven hills': 0.0,
   u'shangri la vegetarian restaurant': 0.0,
   u'singapore malaysian restaurant': 0.0,
   u'skool': 0.0,
   u'so': 0.0,
   u'sotto mare': 0.0,
   u'source': 0.0,
   u'specchio ristorante': 0.0,
   u'spruce': 0.0,
   u'straits restaurant': 0.0,
   u'stroganoff restaurant': 0.0,
   u'sunflower potrero hill': 0.0,
   u'sushi bistro': 0.0,
   u'taiwan restaurant': 0.0,
   u'tanuki restaurant': 0.0,
   u'tataki': 0.0,
   u'tekka japanese restaurant': 0.0,
   u'thai cottage restaurant': 0.0,
   u'thai house express': 0.0,
   u'thai idea vegetarian': 0.0,
   u'thai time restaurant': 0.0,
   u'thanh long': 0.0,
   u'the big 4 restaurant': 0.0,
   u'the blue plate': 0.0,
   u'the house': 0.0,
   u'the richmond': 0.0,
   u'the slanted door': 0.0,
   u'the stinking rose': 0.0,
   u'thep phanom thai restaurant': 0.0,
   u'tommys joynt': 0.0,
   u'toraya japanese restaurant': 0.0,
   u'town hall': 0.0,
   u'trattoria contadina': 0.0,
   u'tu lan': 0.0,
   u'tuba restaurant': 0.0,
   u'u lee restaurant': 0.0,
   u'udupi palace': 0.0,
   u'venticello ristorante': 0.0,
   u'vicoletto': 0.0,
   u'yank sing': 0.0,
   u'yummy yummy': 0.0,
   u'z and y restaurant': 0.0,
   u'zadin': 0.0,
   u'zare at fly trap': 0.0,
   u'zarzuela': 0.0,
   u'zen yai thai restaurant': 0.0,
   u'zuni cafe': 0.0,
   u'zushi puzzle': 0.0},
  u'near': {'**NONE**': 0.13300733496332517,
   u'bayview hunters point': 0.0,
   'dontcare': 0.15859820700896493,
   u'haight': 0.0,
   u'japantown': 0.038712306438467806,
   u'marina cow hollow': 0.0,
   u'mission': 0.0,
   u'nopa': 0.669682151589242,
   u'north beach telegraph hill': 0.0,
   u'soma': 0.0,
   u'union square': 0.0},
  u'price': {'**NONE**': 1.0,
   u'10 dollar': 0.0,
   u'10 euro': 0.0,
   u'11 euro': 0.0,
   u'15 euro': 0.0,
   u'18 euro': 0.0,
   u'20 euro': 0.0,
   u'22 euro': 0.0,
   u'25 euro': 0.0,
   u'26 euro': 0.0,
   u'29 euro': 0.0,
   u'37 euro': 0.0,
   u'6': 0.0,
   u'7': 0.0,
   u'9': 0.0,
   u'between 0 and 15 euro': 0.0,
   u'between 10 and 13 euro': 0.0,
   u'between 10 and 15 euro': 0.0,
   u'between 10 and 18 euro': 0.0,
   u'between 10 and 20 euro': 0.0,
   u'between 10 and 23 euro': 0.0,
   u'between 10 and 30 euro': 0.0,
   u'between 11 and 15 euro': 0.0,
   u'between 11 and 18 euro': 0.0,
   u'between 11 and 22 euro': 0.0,
   u'between 11 and 25 euro': 0.0,
   u'between 11 and 29 euro': 0.0,
   u'between 11 and 35 euro': 0.0,
   u'between 13 and 15 euro': 0.0,
   u'between 13 and 18 euro': 0.0,
   u'between 13 and 24 euro': 0.0,
   u'between 15 and 18 euro': 0.0,
   u'between 15 and 22 euro': 0.0,
   u'between 15 and 26 euro': 0.0,
   u'between 15 and 29 euro': 0.0,
   u'between 15 and 33 euro': 0.0,
   u'between 15 and 44 euro': 0.0,
   u'between 15 and 58 euro': 0.0,
   u'between 18 and 26 euro': 0.0,
   u'between 18 and 29 euro': 0.0,
   u'between 18 and 44 euro': 0.0,
   u'between 18 and 55 euro': 0.0,
   u'between 18 and 58 euro': 0.0,
   u'between 18 and 73 euro': 0.0,
   u'between 18 and 78 euro': 0.0,
   u'between 2 and 15 euro': 0.0,
   u'between 20 and 30 euro': 0.0,
   u'between 21 and 23 euro': 0.0,
   u'between 22 and 29 euro': 0.0,
   u'between 22 and 30 dollar': 0.0,
   u'between 22 and 37 euro': 0.0,
   u'between 22 and 58 euro': 0.0,
   u'between 22 and 73 euro': 0.0,
   u'between 23 and 29': 0.0,
   u'between 23 and 29 euro': 0.0,
   u'between 23 and 37 euro': 0.0,
   u'between 23 and 58': 0.0,
   u'between 23 and 58 euro': 0.0,
   u'between 26 and 33 euro': 0.0,
   u'between 26 and 34 euro': 0.0,
   u'between 26 and 37 euro': 0.0,
   u'between 29 and 37 euro': 0.0,
   u'between 29 and 44 euro': 0.0,
   u'between 29 and 58 euro': 0.0,
   u'between 29 and 73 euro': 0.0,
   u'between 30 and 58': 0.0,
   u'between 30 and 58 euro': 0.0,
   u'between 31 and 50 euro': 0.0,
   u'between 37 and 110 euro': 0.0,
   u'between 37 and 44 euro': 0.0,
   u'between 37 and 58 euro': 0.0,
   u'between 4 and 22 euro': 0.0,
   u'between 4 and 58 euro': 0.0,
   u'between 5 an 30 euro': 0.0,
   u'between 5 and 10 euro': 0.0,
   u'between 5 and 11 euro': 0.0,
   u'between 5 and 15 dollar': 0.0,
   u'between 5 and 20 euro': 0.0,
   u'between 5 and 25 euro': 0.0,
   u'between 6 and 10 euro': 0.0,
   u'between 6 and 11 euro': 0.0,
   u'between 6 and 15 euro': 0.0,
   u'between 6 and 29 euro': 0.0,
   u'between 7 and 11 euro': 0.0,
   u'between 7 and 13 euro': 0.0,
   u'between 7 and 15 euro': 0.0,
   u'between 7 and 37 euro': 0.0,
   u'between 8 and 22 euro': 0.0,
   u'between 9 and 13 dolllar': 0.0,
   u'between 9 and 15 euro': 0.0,
   u'between 9 and 58 euro': 0.0,
   u'bteween 11 and 15 euro': 0.0,
   u'bteween 15 and 22 euro': 0.0,
   u'bteween 22 and 37': 0.0,
   u'bteween 30 and 58 euro': 0.0,
   u'bteween 51 and 73 euro': 0.0,
   u'netween 20 and 30 euro': 0.0},
  u'pricerange': {'**NONE**': 0.22571148184494605,
   u'cheap': 0.0,
   'dontcare': 0.774288518155054,
   u'expensive': 0.0,
   u'moderate': 0.0},
  'requested': {u'addr': 0.0,
   u'allowedforkids': 0.0,
   u'area': 0.0,
   u'food': 0.0,
   u'goodformeal': 0.0,
   u'name': 0.0,
   u'near': 0.0,
   u'phone': 0.0,
   u'postcode': 0.0,
   u'price': 0.0,
   u'pricerange': 0.0}},
 'features': {'inform_info': [False,
   False,
   False,
   True,
   True,
   False,
   False,
   False,
   True,
   True,
   False,
   True,
   False,
   False,
   False,
   False,
   True,
   False,
   False,
   False,
   False,
   True,
   False,
   False,
   False],
  'informedVenueSinceNone': [],
  'lastActionInformNone': False,
  'lastInformedVenue': '',
  'offerHappened': False},
 'userActs': [(u'inform(allowedforkids="1")', 0.90842356395668944),
  (u'inform(allowedforkids="dontcare")', 0.0091759955955221153),
  (u'inform(allowedforkids="0")', 0.0091759955955221153),
  (u'inform(postcode)', 0.025509267755551478),
  (u'inform(area="stonestown")', 0.024683428151954491),
  ('null()', 0.023031748944760511)]}

    b3 = {'beliefs': {u'area': {'**NONE**': 0.12910550615265692,
   u'centre': 0.8338099777773861,
   'dontcare': 0.0,
   u'east': 0.03708451606995696,
   u'north': 0.0,
   u'south': 0.0,
   u'west': 0.0},
  'discourseAct': {u'ack': 0.0,
   'bye': 0.0,
   u'hello': 0.0,
   u'none': 1.0,
   u'repeat': 0.0,
   u'silence': 0.0,
   u'thankyou': 0.0},
  u'food': {'**NONE**': 0.020895546925810415,
   u'afghan': 0.0,
   u'african': 0.0,
   u'afternoon tea': 0.0,
   u'asian oriental': 0.0,
   u'australasian': 0.0,
   u'australian': 0.0,
   u'austrian': 0.0,
   u'barbeque': 0.0,
   u'basque': 0.0,
   u'belgian': 0.0,
   u'bistro': 0.0,
   u'brazilian': 0.0,
   u'british': 0.0,
   u'canapes': 0.0,
   u'cantonese': 0.0,
   u'caribbean': 0.0,
   u'catalan': 0.0,
   u'chinese': 0.0,
   u'christmas': 0.0,
   u'corsica': 0.0,
   u'creative': 0.0,
   u'crossover': 0.0,
   u'cuban': 0.0,
   u'danish': 0.0,
   'dontcare': 0.0,
   u'eastern european': 0.0,
   u'english': 0.0,
   u'eritrean': 0.0,
   u'european': 0.0,
   u'french': 0.0,
   u'fusion': 0.0,
   u'gastropub': 0.0,
   u'german': 0.0,
   u'greek': 0.0,
   u'halal': 0.0,
   u'hungarian': 0.0,
   u'indian': 0.0,
   u'indonesian': 0.0,
   u'international': 0.0,
   u'irish': 0.0,
   u'italian': 0.0,
   u'jamaican': 0.0,
   u'japanese': 0.0,
   u'korean': 0.0,
   u'kosher': 0.0,
   u'latin american': 0.0,
   u'lebanese': 0.0,
   u'light bites': 0.0,
   u'malaysian': 0.0,
   u'mediterranean': 0.9791044530741896,
   u'mexican': 0.0,
   u'middle eastern': 0.0,
   u'modern american': 0.0,
   u'modern eclectic': 0.0,
   u'modern european': 0.0,
   u'modern global': 0.0,
   u'molecular gastronomy': 0.0,
   u'moroccan': 0.0,
   u'new zealand': 0.0,
   u'north african': 0.0,
   u'north american': 0.0,
   u'north indian': 0.0,
   u'northern european': 0.0,
   u'panasian': 0.0,
   u'persian': 0.0,
   u'polish': 0.0,
   u'polynesian': 0.0,
   u'portuguese': 0.0,
   u'romanian': 0.0,
   u'russian': 0.0,
   u'scandinavian': 0.0,
   u'scottish': 0.0,
   u'seafood': 0.0,
   u'singaporean': 0.0,
   u'south african': 0.0,
   u'south indian': 0.0,
   u'spanish': 0.0,
   u'sri lankan': 0.0,
   u'steakhouse': 0.0,
   u'swedish': 0.0,
   u'swiss': 0.0,
   u'thai': 0.0,
   u'the americas': 0.0,
   u'traditional': 0.0,
   u'turkish': 0.0,
   u'tuscan': 0.0,
   u'unusual': 0.0,
   u'vegetarian': 0.0,
   u'venetian': 0.0,
   u'vietnamese': 0.0,
   u'welsh': 0.0,
   u'world': 0.0},
  'method': {u'byalternatives': 0.0,
   u'byconstraints': 0.6359877465366015,
   u'byname': 0.0,
   u'finished': 0.0,
   u'none': 0.0,
   u'restart': 0.0},
  u'name': {'**NONE**': 1.0,
   u'ali baba': 0.0,
   u'anatolia': 0.0,
   u'ask': 0.0,
   u'backstreet bistro': 0.0,
   u'bangkok city': 0.0,
   u'bedouin': 0.0,
   u'bloomsbury restaurant': 0.0,
   u'caffe uno': 0.0,
   u'cambridge lodge restaurant': 0.0,
   u'charlie chan': 0.0,
   u'chiquito restaurant bar': 0.0,
   u'city stop restaurant': 0.0,
   u'clowns cafe': 0.0,
   u'cocum': 0.0,
   u'cote': 0.0,
   u'cotto': 0.0,
   u'curry garden': 0.0,
   u'curry king': 0.0,
   u'curry prince': 0.0,
   u'curry queen': 0.0,
   u'da vince pizzeria': 0.0,
   u'da vinci pizzeria': 0.0,
   u'darrys cookhouse and wine shop': 0.0,
   u'de luca cucina and bar': 0.0,
   u'dojo noodle bar': 0.0,
   u'don pasquale pizzeria': 0.0,
   u'efes restaurant': 0.0,
   u'eraina': 0.0,
   u'fitzbillies restaurant': 0.0,
   u'frankie and bennys': 0.0,
   u'galleria': 0.0,
   u'golden house': 0.0,
   u'golden wok': 0.0,
   u'gourmet burger kitchen': 0.0,
   u'graffiti': 0.0,
   u'grafton hotel restaurant': 0.0,
   u'hakka': 0.0,
   u'hk fusion': 0.0,
   u'hotel du vin and bistro': 0.0,
   u'india house': 0.0,
   u'j restaurant': 0.0,
   u'jinling noodle bar': 0.0,
   u'kohinoor': 0.0,
   u'kymmoy': 0.0,
   u'la margherita': 0.0,
   u'la mimosa': 0.0,
   u'la raza': 0.0,
   u'la tasca': 0.0,
   u'lan hong house': 0.0,
   u'little seoul': 0.0,
   u'loch fyne': 0.0,
   u'mahal of cambridge': 0.0,
   u'maharajah tandoori restaurant': 0.0,
   u'meghna': 0.0,
   u'meze bar restaurant': 0.0,
   u'michaelhouse cafe': 0.0,
   u'midsummer house restaurant': 0.0,
   u'nandos': 0.0,
   u'nandos city centre': 0.0,
   u'panahar': 0.0,
   u'peking restaurant': 0.0,
   u'pipasha restaurant': 0.0,
   u'pizza express': 0.0,
   u'pizza express fen ditton': 0.0,
   u'pizza hut': 0.0,
   u'pizza hut cherry hinton': 0.0,
   u'pizza hut city centre': 0.0,
   u'pizza hut fen ditton': 0.0,
   u'prezzo': 0.0,
   u'rajmahal': 0.0,
   u'restaurant alimentum': 0.0,
   u'restaurant one seven': 0.0,
   u'restaurant two two': 0.0,
   u'rice boat': 0.0,
   u'rice house': 0.0,
   u'riverside brasserie': 0.0,
   u'royal spice': 0.0,
   u'royal standard': 0.0,
   u'saffron brasserie': 0.0,
   u'saigon city': 0.0,
   u'saint johns chop house': 0.0,
   u'sala thong': 0.0,
   u'sesame restaurant and bar': 0.0,
   u'shanghai family restaurant': 0.0,
   u'shiraz restaurant': 0.0,
   u'sitar tandoori': 0.0,
   u'stazione restaurant and coffee bar': 0.0,
   u'taj tandoori': 0.0,
   u'tandoori palace': 0.0,
   u'tang chinese': 0.0,
   u'thanh binh': 0.0,
   u'the cambridge chop house': 0.0,
   u'the copper kettle': 0.0,
   u'the cow pizza kitchen and bar': 0.0,
   u'the gandhi': 0.0,
   u'the gardenia': 0.0,
   u'the golden curry': 0.0,
   u'the good luck chinese food takeaway': 0.0,
   u'the hotpot': 0.0,
   u'the lucky star': 0.0,
   u'the missing sock': 0.0,
   u'the nirala': 0.0,
   u'the oak bistro': 0.0,
   u'the river bar steakhouse and grill': 0.0,
   u'the slug and lettuce': 0.0,
   u'the varsity restaurant': 0.0,
   u'travellers rest': 0.0,
   u'ugly duckling': 0.0,
   u'venue': 0.0,
   u'wagamama': 0.0,
   u'yippee noodle bar': 0.0,
   u'yu garden': 0.0,
   u'zizzi cambridge': 0.0},
  u'pricerange': {'**NONE**': 0.1340777132648503,
   u'cheap': 0.0,
   'dontcare': 0.8659222867351497,
   u'expensive': 0.0,
   u'moderate': 0.0},
  'requested': {u'addr': 0.0,
   u'area': 0.0,
   u'description': 0.0,
   u'food': 0.0,
   u'name': 0.0,
   u'phone': 0.0,
   u'postcode': 0.0,
   u'pricerange': 0.0,
   u'signature': 0.0}},
 'features': {'inform_info': [False,
   False,
   True,
   False,
   True,
   False,
   False,
   True,
   False,
   False,
   False,
   False,
   True,
   False,
   False,
   False,
   False,
   True,
   False,
   False,
   False,
   False,
   True,
   False,
   False],
  'informedVenueSinceNone': [],
  'lastActionInformNone': False,
  'lastInformedVenue': '',
  'offerHappened': False},
 'userActs': [(u'inform(food="mediterranean")', 0.84415346579983519),
  (u'inform(area="east")', 0.037084516069956962),
  ('null()', 0.048530354363153554),
  ('reqmore()', 0.04541708634740408),
  (u'confirm(phone)', 0.024814577419650211)]}

    return b1, b2, b3


def main():
    """
    unit test
    :return:
    """

    Settings.init('config/Tut-gp-Multidomain.cfg', 12345)
    Ontology.init_global_ontology()

    b1, b2, b3 = get_test_beliefs()
    '''state1 = DIP_state(b1, domainString='SFRestaurants')
    state2 = DIP_state(b2, domainString='SFRestaurants')
    state3 = DIP_state(b3, domainString='CamRestaurants')'''
    state1 = padded_state(b1, domainString='SFRestaurants')
    state2 = padded_state(b2, domainString='SFRestaurants')
    state3 = padded_state(b3, domainString='CamRestaurants')
    print state1.get_beliefStateVec('area')[:state1.max_v]
    print len(state2.get_beliefStateVec('near'))-state2.max_v
    print len(state3.get_beliefStateVec('pricerange'))-state3.max_v
    #print len(state3.get_beliefStateVec('general'))
    s2 = state2.get_beliefStateVec('food')
    s3 = state3.get_beliefStateVec('food')
    a=1
    #print state3.get_beliefStateVec('general')[:state2.max_v]
    #print state2.max_v
    #print state3.max_v


if __name__ == '__main__':
    main()

