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
UserModel.py - goal, agenda inventor for sim user 
====================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 
    
    import :mod:`utils.DiaAct` |.|
    import :mod:`utils.dact` |.|
    import :mod:`usersimulator.UMHdcSim` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger`

************************

'''

__author__ = "cued_dialogue_systems_group"
import copy

from utils import DiaAct, dact, Settings, ContextLogger
import UMHdcSim
from ontology import Ontology 
logger = ContextLogger.getLogger('')


class UMAgenda(object):
    '''Agenda of :class:`DiaAct` acts corresponding to a goal.
    
    :param domain: domain tag (ie CamRestaurants)
    :type domain: str
    '''
    def __init__(self, dstring):
        self.dstring = dstring
        self.agenda_items = []  # Stack of DiaAct
        self.rules = None
        # HDC probs on how simuser will conditionally behave: values are upper limits of ranges uniform sample falls in
        self.NOT_MENTION = 1.0 # default - this value doesnt actually matter
        self.LEAVE_INFORM = 0.4
        self.CONFIRM = -1.0  # doesn't make sense actually to change inform(X=Y) to a confirm() act.
        # ONE OTHER OPTION IS TO APPEND constraint into another inform act so it is inform(X=Y,A=B)

    def init(self, goal):
        """
        Initialises the agenda by creating DiaActs corresponding to the
        constraints in the goal G. Uses the default order for the
        dialogue acts on the agenda: an inform act is created for
        each constraint. Finally a bye act is added at the bottom of the agenda.
        
        :param goal: 
               
        .. Note::
            No requests are added to the agenda.
        """
        self.agenda_items = []
        self.append_dact_to_front(DiaAct.DiaAct('inform(type="%s")' % goal.request_type))

        for const in goal.constraints:
            slot = const.slot
            value = const.val
            if slot == 'method':
                continue

            do_not_add = False
            dia_act = DiaAct.DiaAct('inform()')
            slots = Ontology.global_ontology.getSlotsToExpress(self.dstring, slot, value)

            for s in slots:
                val = goal.get_correct_const_value(s)
                if val is not None:
                    dia_act.append(slot, val)
                if len(slots) == 1 and self.contains(s, val):
                    do_not_add = True
                elif len(slots) > 1:
                    # Slot value pair might already be in other act on agenda: remove that one
                    self.filter_acts_slot(s)

            if not do_not_add:
                self.append_dact_to_front(dia_act)

        # CONDITIONALLY INIT THE AGENDA:
        if len(goal.copied_constraints): # dont need self.CONDITIONAL_BEHAVIOUR - list will be empty as appropriate
            self._conditionally_init_agenda(goal)
          
        # Finally append a bye() act to complete agenda:
        self.append_dact_to_front(DiaAct.DiaAct('bye()'))
        return

    def _conditionally_init_agenda(self, goal):
        """Use goal.copied_constraints -- to conditionally init Agenda.
        Probabilistically remove/alter the agenda for this constraint then:

        :param:
        :returns:
        """

        for dact in goal.copied_constraints:
            #print dact.slot, dact.op, dact.val   # TODO - delete debug prints
            uniform_sample = Settings.random.uniform()
            #print uniform_sample
            if uniform_sample < self.CONFIRM:
                # TODO - remove/change this - 
                # Decided this doesn't make sense - (so set self.CONFIRM=0) - info is in belief state, that is enough.
                logger.info("changing inform() act to a confirm() act")
                self.replace_acts_slot(dact.slot, replaceact="confirm") 
            elif uniform_sample < self.LEAVE_INFORM:
                pass
            else:
                logger.info("removing the inform() act and not replacing with anything.") 
                self.filter_acts_slot(dact.slot)
        return


    def contains(self, slot, value, negate=False):
        '''Check if slot, value pair is in an agenda dialogue act

        :param slot:
        :param value:
        :param negate: None
        :type negate: bool
        :returns: (bool) slot, value is in an agenda dact?
        '''
        for dact in self.agenda_items:
            if dact.contains(slot, value, negate):
                return True
        return False

    def get_agenda_with_act(self, act):
        '''agenda items with this act
        :param act: dialogue act 
        :type act: str
        :return: (list) agenda items
        '''
        items = []
        for ait in self.agenda_items:
            if ait.act == act:
                items.append(ait)
        return items

    def get_agenda_with_act_slot(self, act, slot):
        '''
        :param act: dialogue act
        :type act: str
        :param slot: slot name
        :type slot: str
        :return: (list) of agenda items
        '''
        items = []
        for ait in self.agenda_items:
            if ait.act == act:
                for item in ait.items:
                    if item.slot == slot:
                        items.append(ait)
                        break
        return items

    def replace_acts_slot(self, slot, replaceact="confirm"):
        """
        """
        for ait in self.agenda_items:
            if len(ait.items) == 1:
                if ait.items[0].slot == slot:
                    print ait.act
                    print ait.items
                    ait.act = replaceact
                    print ait
                    raw_input('going to change this to confirm')


    def filter_acts_slot(self, slot):
        '''
        Any acts related to the given slot are removed from the agenda.
        :param slot: slot name
        :type slot: str
        :return: None
        '''
        deleted = []
        for ait in self.agenda_items:
            if ait.act in ['inform', 'confirm', 'affirm'] and len(ait.items) > 0:
                if len(ait.items) > 1:
                    pass
#                     logger.error('Assumes all agenda items have only one semantic items: {}'.format(ait))
                for only_item in ait.items:
                    if only_item.slot == slot:
                        deleted.append(ait)

        for ait in deleted:
            self.agenda_items.remove(ait)

    def filter_constraints(self, dap):
        '''Filters out acts on the agenda that convey the constraints mentioned in the given dialogue act. 
        Calls :meth:`filter_acts_slot` to do so. 

        :param dap:
        :returns: None
        '''
        if dap.act in ['inform', 'confirm'] or \
            (dap.act in ['affirm', 'negate'] and len(dap.items) > 0):
            for item in dap.items:
                self.filter_acts_slot(item.slot)

    def size(self):
        '''Utility func to get size of agenda_items list
        
        :returns: (int) length
        '''
        return len(self.agenda_items)

    def clear(self):
        '''
        Erases all acts on the agenda (empties list)
        
        :return: None
        '''
        self.agenda_items = []

    def append_dact_to_front(self, dact):
        '''Adds the given dialogue act to the front of the agenda

        :param (instance): dact
        :returns: None
        '''
        self.agenda_items = [dact] + self.agenda_items

    def push(self, dact):
        # if dact.act == 'null':
        #     logger.warning('null() in agenda')
        # if dact.act == 'bye' and len(self.agenda_items) > 0:
        #     logger.warning('bye() in agenda')
        self.agenda_items.append(dact)

    def pop(self):
        return self.agenda_items.pop()

    def remove(self, dact):
        self.agenda_items.remove(dact)


class UMGoal(object):
    '''Defines a goal within a domain

    :param patience: user patience 
    :type patience: int
    '''
    def __init__(self, patience, domainString):
        self.constraints = []
        self.copied_constraints = []  # goals copied over from other domains. Used to conditionally create agenda.
        self.requests = {}
        self.prev_slot_values = {}
        self.patience = patience
        self.request_type = Ontology.global_ontology.get_type(domainString)  

        self.system_has_informed_name_none = False
        self.no_relaxed_constraints_after_name_none = False

    def clear(self, patience, domainString):
        self.constraints = []
        self.requests = {}
        self.prev_slot_values = {}
        self.patience = patience
        self.request_type = Ontology.global_ontology.get_type(domainString)

        self.system_has_informed_name_none = False
        self.no_relaxed_constraints_after_name_none = False

    '''
    Methods for constraints.
    '''
    def set_copied_constraints(self, all_conditional_constraints):
        """Creates a list of dacts, where the constraints have come from earlier domains in the dialog.

        :param all_conditional_constraints: of all previous constraints (over all domains in dialog)
        :type all_conditional_constraints: dict
        :returns: None
        """
        for dact in self.constraints:
            slot,op,value = dact.slot,dact.op,dact.val
            if slot in all_conditional_constraints.keys():
                if len(all_conditional_constraints[slot]):
                    if value in all_conditional_constraints[slot]:
                        self.copied_constraints.append(dact)
        return

    def add_const(self, slot, value, negate=False):
        """
        """
        if not negate:
            op = '='
        else:
            op = '!='
        item = dact.DactItem(slot, op, value)
        self.constraints.append(item)

    def replace_const(self, slot, value, negate=False):
        self.remove_slot_const(slot, negate)
        self.add_const(slot, value, negate)

    def contains_slot_const(self, slot):
        for item in self.constraints:
            # an error introduced here by dact.py __eq__ method: 
            #if item.slot == slot:
            if str(item.slot) == slot:
                return True
        return False

    def remove_slot_const(self, slot, negate=None):
        copy_consts = copy.deepcopy(self.constraints)
        
        if negate is not None:
            if not negate:
                op = '='
            else:
                op = '!='
        
            for item in copy_consts:
                if item.slot == slot:
                    if item.op == op:
                        self.constraints.remove(item)
        else:
            for item in copy_consts:
                if item.slot == slot:
                    self.constraints.remove(item)

    def get_correct_const_value(self, slot, negate=False):
        '''
        :return: (list of) value of the given slot in user goal constraint.

        '''
        values = []
        for item in self.constraints:
            if item.slot == slot:
                if item.op == '!=' and negate or item.op == '=' and not negate:
                    values.append(item.val)

        if len(values) == 1:
            return values[0]
        elif len(values) == 0:
            return None
        logger.error('Multiple values are found for %s in constraint: %s' % (slot, str(values)))
        return values

    def get_correct_const_value_list(self, slot, negate=False):
        '''
        :return: (list of) value of the given slot in user goal constraint.
        '''
        values = []
        for item in self.constraints:
            if item.slot == slot:
                if (item.op == '!=' and negate) or (item.op == '=' and not negate):
                    values.append(item.val)
        return values

    def add_prev_used(self, slot, value):
        '''
        Adds the given slot-value pair to the record of previously used slot-value pairs.
        '''
        if slot not in self.prev_slot_values:
            self.prev_slot_values[slot] = set()
        self.prev_slot_values[slot].add(value)

    def add_name_constraint(self, value, negate=False):
        if value in [None, 'none']:
            return

        wrong_venues = self.get_correct_const_value_list('name', negate=True)
        correct_venue = self.get_correct_const_value('name', negate=False)

        if not negate:
            # Adding name=value but there is name!=value.
            if value in wrong_venues:
                logger.error('Failed to add name=%s: already got constraint name!=%s.' %
                             (value, value))
                return
            
            # Can have only one name= constraint.
            if correct_venue is not None:
                #logger.debug('Failed to add name=%s: already got constraint name=%s.' %
                #             (value, correct_venue))
                self.replace_const('name', value) # ic340: added to override previously informed venues, to avoid
                                                    # simuser getting obsessed with a wrong venue
                return

            # Adding name=value, then remove all name!=other.
            self.replace_const('name', value)
            return

        # if not negate and not self.is_suitable_venue(value):
        #     logger.debug('Failed to add name=%s: %s is not a suitable venue for goals.' % (value, value))
        #     return

        if negate:
            # Adding name!=value but there is name=value.
            if correct_venue == value:
                logger.error('Failed to add name!=%s: already got constraint name=%s.' % (value, value))
                return

            # Adding name!=value, but there is name=other. No need to add.
            if correct_venue is not None:
                return

            self.add_const('name', value, negate=True)
            return

    # def is_correct(self, item):
    #     '''
    #     Check if the given items are correct in goal constraints.
    #     :param item: set[(slot, op, value), ...]
    #     :return:
    #     '''
    #     if type(item) is not set:
    #         item = set([item])
    #     for it in item:
    #         for const in self.constraints:
    #             if const.match(it):

    def is_satisfy_all_consts(self, item):
        '''
        Check if all the given items set[(slot, op, value),..] satisfies all goal constraints (conjunction of constraints).
        '''
        if type(item) is not set:
            item = set([item])
        for it in item:
            for const in self.constraints:
                if not const.match(it):
                    return False
        return True

    def is_completed(self):
        # If the user has not specified any constraints, return True
        if not self.constraints:
            return True
        if (self.system_has_informed_name_none and not self.no_relaxed_constraints_after_name_none) or\
                (self.is_venue_recommended() and self.are_all_requests_filled()):
            return True
        return False

    '''
    Methods for requests.
    '''
    def reset_requests(self):
        for info in self.requests:
            self.requests[info] = None

    def fill_requests(self, dact_items):
        for item in dact_items:
            if item.op != '!=':
                self.requests[item.slot] = item.val

    def are_all_requests_filled(self):
        '''
        :return: True if all request slots have a non-empty value.
        '''
        return None not in self.requests.values()

    def is_venue_recommended(self):
        '''
        Returns True if the request slot 'name' is not empty.
        :return:
        '''
        if 'name' in self.requests and self.requests['name'] is not None:
            return True
        return False

    def get_unsatisfied_requests(self):
        results = []
        for info in self.requests:
            if self.requests[info] is None:
                results.append(info)
        return results

    def __str__(self):
        result = 'constraints: ' + str(self.constraints) + '\n'
        result += 'requests:    ' + str(self.requests) + '\n'
        if self.patience is not None:
            result += 'patience:    ' + str(self.patience) + '\n'
        return result


class GoalGenerator(object):
    '''
    Master class for defining a goal generator to generate domain specific goals for the simulated user.
    
    This class also defines the interface for new goal generators.
    
    To implement domain-specific behaviour, derive from this class and override init_goals.
    '''
    def __init__(self, dstring):
        '''
        The init method of the goal generator reading the config and setting the default parameter values.
        
        :param dstring: domain name tag
        :type dstring: str
        '''
        self.dstring = dstring
        configlist = []
        
        self.CONDITIONAL_BEHAVIOUR = False
        if Settings.config.has_option("conditional","conditionalsimuser"):
            self.CONDITIONAL_BEHAVIOUR = Settings.config.getboolean("conditional","conditionalsimuser")
        
        self.MAX_VENUES_PER_GOAL = 4
        if Settings.config.has_option('goalgenerator','maxvenuespergoal'):
            configlist.append('maxvenuespergoal')
            self.MAX_VENUES_PER_GOAL = Settings.config.getint('goalgenerator','maxvenuespergoal')
        self.MIN_VENUES_PER_GOAL = 1

        self.MAX_CONSTRAINTS = 3
        if Settings.config.has_option('goalgenerator','maxconstraints'):
            configlist.append('maxconstraints')
            self.MAX_CONSTRAINTS = Settings.config.getint('goalgenerator','maxconstraints')

        self.MAX_REQUESTS = 3
        if Settings.config.has_option('goalgenerator','maxrequests'):
            configlist.append('maxrequests')
            self.MAX_REQUESTS = Settings.config.getint('goalgenerator','maxrequests')
        self.MIN_REQUESTS = 1
        if Settings.config.has_option('goalgenerator','minrequests'):
            configlist.append('minrequests')
            self.MIN_REQUESTS = Settings.config.getint('goalgenerator','minrequests')
        assert(self.MIN_REQUESTS > 0)
        if Settings.config.has_option('goalgenerator','minvenuespergoal'):
            self.MIN_VENUES_PER_GOAL = int(Settings.config.get('goalgenerator','minvenuespergoal'))
#        self.PERCENTAGE_OF_ZERO_SOLUTION_TASKS = 50
#        if Settings.config.has_option('GOALGENERATOR','PERCZEROSOLUTION'):
#            self.PERCENTAGE_OF_ZERO_SOLUTION_TASKS = int(Settings.config.get('GOALGENERATOR','PERCZEROSOLUTION'))
#        self.NO_REQUESTS_WITH_VALUE_NONE = False
#        if Settings.config.has_option('GOALGENERATOR','NOREQUESTSWITHVALUENONE'):
#            self.NO_REQUESTS_WITH_VALUE_NONE

        if self.MIN_VENUES_PER_GOAL > self.MAX_VENUES_PER_GOAL:
            logger.error('Invalid config: MIN_VENUES_PER_GOAL > MAX_VENUES_PER_GOAL')
        
        self.conditional_constraints = []
        self.conditional_constraints_slots = []
        
       
    def init_goal(self, otherDomainsConstraints, um_patience):
        '''
        Initialises the goal g with random constraints and requests

        :param otherDomainsConstraints: of constraints from other domains in this dialog which have already had goals generated.
        :type otherDomainsConstraints: list
        :param um_patience: the patiance value for this goal
        :type um_patience: int
        :returns: (instance) of :class:`UMGoal`
        '''
        
        # clean/parse the domainConstraints list - contains other domains already generated goals:
        self._set_other_domains_constraints(otherDomainsConstraints)

        # Set initial goal status vars
        goal = UMGoal(um_patience, domainString=self.dstring)
        logger.debug(str(goal))
        num_attempts_to_resample = 2000
        while True:
            num_attempts_to_resample -= 1
            # Randomly sample a goal (ie constraints):
            self._init_consts_requests(goal, um_patience)
            # Check that there are venues that satisfy the constraints:
            venues = Ontology.global_ontology.entity_by_features(self.dstring, constraints=goal.constraints)
            #logger.info('num_venues: %d' % len(venues))
            if self.MIN_VENUES_PER_GOAL < len(venues) < self.MAX_VENUES_PER_GOAL:
                break
            if num_attempts_to_resample == 0:
                logger.warning('Maximum number of goal resampling attempts reached.')
                
        if self.CONDITIONAL_BEHAVIOUR:
            # now check self.generator.conditional_constraints list against self.goal -assume any values that are the same
            # are because they are conditionally copied over from earlier domains goal. - set self.goal.copied_constraints
            goal.set_copied_constraints(all_conditional_constraints=self.conditional_constraints)

        # logger.warning('SetSuitableVenues is deprecated.')
        return goal
        
    def _set_other_domains_constraints(self, otherDomainsConstraints):
        """Simplest approach for now: just look for slots with same name 
        """
        # Get a list of slots that are valid for this task.
        valid_const_slots = Ontology.global_ontology.getValidSlotsForTask(self.dstring) 
        self.conditional_constraints = {slot: [] for slot in valid_const_slots}

        if not self.CONDITIONAL_BEHAVIOUR:
            self.conditional_constraint_slots = []
            return

        for const in otherDomainsConstraints:
            if const.slot in valid_const_slots and const.val != "dontcare": #TODO think dontcare should be dealt with diff 
                # issue is that first domain may be dontcare - but 2nd should be generated conditioned on first.
                if const.op == "!=":
                    continue
                
                #TODO delete: if const.val in Ontology.global_ontology.ontology['informable'][const.slot]: #make sure value is valid for slot
                if Ontology.global_ontology.is_value_in_slot(self.dstring, value=const.val, slot=const.slot):
                    self.conditional_constraints[const.slot] += [const.val]  
        self.conditional_constraint_slots = [s for s,v in self.conditional_constraints.iteritems() if len(v)]
        return


    def _init_consts_requests(self, goal, um_patience):
        '''
        Randomly initialises constraints and requests of the given goal.
        '''
        goal.clear(um_patience, domainString=self.dstring)

        # Randomly pick a task: bar, hotel, or restaurant (in case of TownInfo)
        goal.request_type = Ontology.global_ontology.getRandomValueForSlot(self.dstring, slot='type')

        # Get a list of slots that are valid for this task.
        valid_const_slots = Ontology.global_ontology.getValidSlotsForTask(self.dstring)
        
        # First randomly sample some slots from those that are valid: 
        sampling_probs = Ontology.global_ontology.get_sample_prob(self.dstring, 
                                                                  candidate=valid_const_slots,
                                                                  conditional_values=self.conditional_constraint_slots)
        random_slots = list(Settings.random.choice(valid_const_slots,
                                size=min(self.MAX_CONSTRAINTS, len(valid_const_slots)),
                                replace=False,
                                p=sampling_probs))


        # Now randomly fill in some constraints for the sampled slots:
        for slot in random_slots:
            goal.add_const(slot, Ontology.global_ontology.getRandomValueForSlot(self.dstring, slot=slot, 
                                                            nodontcare=False,
                                                            conditional_values=self.conditional_constraints[slot]))
        
        # Add requests. Assume that the user always wants to know the name of the place
        goal.requests['name'] = None
        
        if self.MIN_REQUESTS == self.MAX_REQUESTS:
            n = self.MIN_REQUESTS -1  # since 'name' is already included
        else:
            n = Settings.random.randint(low=self.MIN_REQUESTS-1,high=self.MAX_REQUESTS)
        valid_req_slots = Ontology.global_ontology.getValidRequestSlotsForTask(self.dstring)
        if n > 0 and len(valid_req_slots) >= n:   # ie more requests than just 'name'
            choosen = Settings.random.choice(valid_req_slots, n,replace=False)
            for reqslot in choosen:
                goal.requests[reqslot] = None


class UM(object):
    '''Simulated user
    
    :param None:
    '''
    def __init__(self, domainString):
        self.max_patience = 5
        self.sample_patience = None
        if Settings.config.has_option('goalgenerator', 'patience'): # only here for backwards compatibility; should actually be in um
            self.max_patience = Settings.config.get('goalgenerator', 'patience')
        if Settings.config.has_option('usermodel', 'patience'):
            self.max_patience = Settings.config.get('usermodel', 'patience')
        if isinstance(self.max_patience,str) and len(self.max_patience.split(',')) > 1:
            self.sample_patience = [int(x.strip()) for x in self.max_patience.split(',')]
            if len(self.sample_patience) != 2 or self.sample_patience[0] > self.sample_patience[1]:
                logger.error('Patience should be either a single int or a range between 2 ints.')
        else:
            self.max_patience = int(self.max_patience)
        
        self.patience_old_style = False
        if Settings.config.has_option('usermodel', 'oldstylepatience'):
            self.patience_old_style = Settings.config.getboolean('usermodel', 'oldstylepatience')
        self.old_style_parameter_sampling = True
        if Settings.config.has_option('usermodel', 'oldstylesampling'):
            self.old_style_parameter_sampling = Settings.config.getboolean('usermodel', 'oldstylesampling')
            
        self.sampleParameters = False
        if Settings.config.has_option('usermodel', 'sampledialogueprobs'):
            self.sampleParameters = Settings.config.getboolean('usermodel', 'sampledialogueprobs')
            
        self.generator = self._load_domain_goal_generator(domainString)
        self.goal = None
        self.prev_goal = None
        self.hdcSim = self._load_domain_simulator(domainString)
        self.lastUserAct = None
        self.lastSysAct = None
        
        
    def init(self, otherDomainsConstraints):
        '''
        Initialises the simulated user.
        1. Initialises the goal G using the goal generator.
        2. Populates the agenda A using the goal G.
        Resets all UM status to their defaults.

        :param otherDomainsConstraints: of domain goals/constraints (slot=val) from other domains in dialog for which goal has already been generated.
        :type otherDomainsConstraints: list
        :returns None:
        '''
        if self.sampleParameters:
            self._sampleParameters()

        if self.sample_patience:
            self.max_patience = Settings.random.randint(self.sample_patience[0], self.sample_patience[1])
        
        self.goal = self.generator.init_goal(otherDomainsConstraints, self.max_patience)
        logger.debug(str(self.goal))

        self.lastUserAct = None
        self.lastSysAct = None
        self.hdcSim.init(self.goal, self.max_patience)  #uses infor in self.goal to do conditional generation of agenda as well.

    def receive(self, sys_act):
        '''
        This method is called to transmit the machine dialogue act to the user.
        It updates the goal and the agenda.
        :param sys_act: System action.
        :return:
        '''
        # Update previous goal.
        self.prev_goal = copy.deepcopy(self.goal)

        # Update the user patience level.
        if self.lastUserAct is not None and self.lastUserAct.act == 'repeat' and\
                        self.lastSysAct is not None and self.lastSysAct.act == 'repeat' and\
                        sys_act.act == 'repeat':
            # Endless cycle of repeating repeats: reduce patience to zero.
            logger.info("Endless cycle of repeating repeats. Setting patience to zero.")
            self.goal.patience = 0

        elif sys_act.act == 'badact' or sys_act.act == 'null' or\
                (self.lastSysAct is not None and self.lastUserAct.act != 'repeat' and self.lastSysAct == sys_act):
            # Same action as last turn. Patience decreased.
            self.goal.patience -= 1
        elif self.patience_old_style:
            # not same action as last time so patience is restored
            self.goal.patience = self.max_patience

        if self.goal.patience < 1:
            logger.debug(str(self.goal))
            logger.debug('All patience gone. Clearing agenda.')
            self.hdcSim.agenda.clear()
            # Pushing bye act onto agenda.
            self.hdcSim.agenda.push(DiaAct.DiaAct('bye()'))
            return

        # Update last system action.
        self.lastSysAct = sys_act

        # Process the sys_act
        self.hdcSim.receive(sys_act, self.goal)

        # logger.warning('should update goal information')

    def respond(self):
        '''
        This method is called after receive() to get the user dialogue act response.
        The method first increments the turn counter, then pops n items off the agenda to form
        the response dialogue act. The agenda and goal are updated accordingly.

        :param None:
        :returns: (instance) of :class:`DiaAct` 
        '''
        user_output = self.hdcSim.respond(self.goal)

        if user_output.act == 'request' and len(user_output.items) > 0:
            # If there is a goal constraint on near, convert act type to confirm
            # TODO-  do we need to add more domain dependent type rules here for other domains beyond CamRestaurants?
            # this whole section mainly seems to stem from having "area" and "near" in ontology... 
            if user_output.contains_slot('near') and self.goal.contains_slot_const('near'):
                for const in self.goal.constraints:
                    if const.slot == 'near':
                        near_const = const.val
                        near_op = const.op
                        break
                # original: bug. constraints is a list --- near_const = self.goal.constraints['near']
                if near_const != 'dontcare':
                    if near_op == "=":   # should be true for 'dontcare' value
                        #TODO - delete-WRONG-user_output.dact['act'] = 'confirm'
                        #TODO-delete-user_output.dact['slots'][0].val = near_const
                        user_output.act = 'confirm'
                        user_output.items[0].val = near_const
                    
        self.lastUserAct = user_output
        #self.goal.update_au(user_output)
        return user_output
    
    def _sampleParameters(self):
        if not self.old_style_parameter_sampling:
            self.max_patience = Settings.random.randint(2,10)
            
    def _load_domain_goal_generator(self, domainString):
        '''
        Loads and instantiates the respective goal generator object as configured in config file. The new object is returned.
        
        Default is GoalGenerator.
        
        .. Note:
            To dynamically load a class, the __init__() must take two arguments: domainString (str), conditional_behaviour (bool)
        
        :param domainString: the domain the goal generator will be loaded for.
        :type domainString: str
        :returns: goal generator object
        '''
        
        generatorClass = None
        
        if Settings.config.has_option('usermodel_' + domainString, 'goalgenerator'):
            generatorClass = Settings.config.get('usermodel_' + domainString, 'goalgenerator')
        
        if generatorClass is None:
            return GoalGenerator(domainString)
        else:
            try:
                # try to view the config string as a complete module path to the class to be instantiated
                components = generatorClass.split('.')
                packageString = '.'.join(components[:-1]) 
                classString = components[-1]
                mod = __import__(packageString, fromlist=[classString])
                klass = getattr(mod, classString)
                return klass(domainString)
            except ImportError:
                logger.error('Unknown domain ontology class "{}" for domain "{}"'.format(generatorClass, domainString))
                
    def _load_domain_simulator(self, domainString):
        '''
        Loads and instantiates the respective simulator object as configured in config file. The new object is returned.
        
        Default is UMHdcSim.
        
        .. Note:
            To dynamically load a class, the __init__() must take one argument: domainString (str)
        
        :param domainString: the domain the simulator will be loaded for.
        :type domainString: str
        :returns: simulator object
        '''
        
        simulatorClass = None
        
        if Settings.config.has_option('usermodel_' + domainString, 'usersimulator'):
            simulatorClass = Settings.config.get('usermodel_' + domainString, 'usersimulator')
        
        if simulatorClass is None:
            return UMHdcSim.UMHdcSim(domainString)
        else:
            try:
                # try to view the config string as a complete module path to the class to be instantiated
                components = simulatorClass.split('.')
                packageString = '.'.join(components[:-1]) 
                classString = components[-1]
                mod = __import__(packageString, fromlist=[classString])
                klass = getattr(mod, classString)
                return klass(domainString)
            except ImportError:
                logger.error('Unknown domain ontology class "{}" for domain "{}"'.format(simulatorClass, domainString))
                
                

#END OF FILE
