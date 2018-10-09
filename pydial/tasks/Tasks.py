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
Tasks.py - Task generator and reader classes
====================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.ContextLogger` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`usersimulator.SimulatedUsersManager`
    
    
Functionality:
    - save seed, with same seed regenerate the same tasks files(s)
    - needs to be used by: 
        -- Camdial - to display tasks to MTurkers
        -- Voiphub - to score by objective measure (RNN doesn't need to know task here)
    - DTMF codes and valid task ids:  

How to generate tasks:

>>> python tasks/Tasks.py -c config/tasks.cfg

This will write two files to the *savename* config option under [tasks] given in the config file tasks.cfg. 
One file (ending .json) is for DialogueServer to read from (set config option *tasksfile* under [dialogueserver]). 
The second file (ending _camdial.json) is for displaying tasks on Camdial - It may end up containing a natural language *translation*
of the task. 


Reading tasks is done via the Tasks.TaskReader() class. This is used by DialogueServer to read the file (ending .json) generated above.

************************

'''


__author__ = "cued_dialogue_systems_group"

import json
import argparse
try:
    from utils import Settings, ContextLogger
    from usersimulator import SimulatedUsersManager
    from ontology import Ontology
except ImportError:
    import os,sys
    curdir = os.path.dirname(os.path.realpath(__file__))
    curdir = curdir.split('/')
    curdir = '/'.join(curdir[:-1]) +'/'
    sys.path.append(curdir)
    from utils import Settings, ContextLogger
    from usersimulator import SimulatedUsersManager
    from ontology import Ontology
logger = ContextLogger.getLogger('')


class TaskCreator(object):
    """
    """
    def __init__(self):
        """
        needs to set:
            - num tasks
            - domains
            - conditional goals/constraints (for multi domain tasks)
            - outfile root name - for .json (to be used by voiphub) and for .xml for Camdial
            - NOTE: not sure if any problems could arise from unicode v. string differences...
        """
        
        # Set number of goals/different dialogues to generate:
        self.MIN_TASK_NUM = 10000   # start the tasks from a large number - so that DTMF = 3*task_id can work
        self.NUM_TASKS = 100
        if Settings.config.has_option("tasks","numtasks"):
            self.NUM_TASKS = Settings.config.getint("tasks","numtasks")

        # SAVE NAME:
        if not Settings.config.has_option("tasks","savename"):
            logger.error("No name given to save file under")
        self.savename = Settings.config.get("tasks","savename")
        # if savedir given, then prepend it to save name
        if Settings.config.has_option("tasks","savedir"):
            directory = Settings.config.get("tasks","savedir")
            self.savename = directory+self.savename

        ## Create goal generators for each domain:
        self.multi_domain_sim_user = SimulatedUsersManager.SimulatedUsersManager(Ontology.global_ontology.possible_domains)
        
        # data structure to hold and write out generated tasks:
        self.tasks = {"possible_domains":Ontology.global_ontology.possible_domains,
                      "tasks": {},    # will add to this, keys=task numbers
                     }


    def _create(self):
        """
        """
        for taskNo in range(self.NUM_TASKS):
            # print "Creating TaskNo: "+str(taskNo)
            # 1. Generate goals
            self.multi_domain_sim_user.restart()
            # 2. format goals into a single task:
            task = dict.fromkeys(self.multi_domain_sim_user.using_domains)
            for dstring in task.keys():
                task[dstring] = {}
                task[dstring]["Cons"] = self._parse_constraints(
                                self.multi_domain_sim_user.simUserManagers[dstring].um.goal.constraints)
                task[dstring]["Reqs"] = self._parse_requests(
                                self.multi_domain_sim_user.simUserManagers[dstring].um.goal.requests)
                task[dstring]["Ents"] = self._get_entities(dstring,
                                            self.multi_domain_sim_user.simUserManagers[dstring].um.goal.constraints) 
                task[dstring]["Pati"] = self.multi_domain_sim_user.simUserManagers[dstring].um.goal.patience
                task[dstring]["Type"] = self.multi_domain_sim_user.simUserManagers[dstring].um.goal.request_type

            # 3. Record goals against task_id
            self.tasks["tasks"][taskNo+self.MIN_TASK_NUM] = task

    def _parse_requests(self, requests):
        """
        """
        return ", ".join(requests.keys())

    def _parse_constraints(self, constraints):
        """
        """
        tmp = []
        for c in constraints:
            tmp.append(str(c.slot+c.op+c.val)) 
        return ", ".join(tmp)

    def _get_entities(self, dstring, constraints):
        """
        """
        ents = Ontology.global_ontology.entity_by_features(dstring, constraints)
        names = []
        if len(ents):
            for ent in ents:
                names.append(ent["name"])
        return ", ".join(names)


    def _write(self):
        """
        """
        #1. write output to VOICEHUB/VOIPHUB file
        with open(self.savename + ".json", "w") as f:
            f.write(json.dumps(self.tasks, indent=4, sort_keys=True))
        #2. write output to CAMDIAL file - could change the Camdial pages format, or else reformat here into XML.
        #   either way still need and extra file with the natural language rendering of the tasks
        self._write_Camdial_version()

    def _form_NL_given_task(self, task):
        """task is a dict - add some "domain_text" keys within domains and a summary "text" covering all domains in task 
        """
        for dstring, dtask in task.iteritems():
            # TODO
            dtask["domain_text"] = self._form_domain_NL_given_task(domain=dstring, task=dtask)
            #print dstring
            #raw_input(dtask)
        task["text"] = self._form_summary_NL(task) 
        return

    def _form_domain_NL_given_task(self, domain, task):
        """
        """
        pass #TODO - note perl script which has template sentences for many domains

    def _form_summary_NL(self, task):
        """
        """
        pass #TODO


    def _write_Camdial_version(self):
        """
        """
        #1. add NATURAL LANGUAGE
        for task in self.tasks["tasks"].itervalues():
            self._form_NL_given_task(task)

        #2. write
        with open(self.savename + "_camdial.json", "w") as f:
            f.write(json.dumps(self.tasks, indent=4, sort_keys=True))


class TaskReader(object):
    """
    """
    def __init__(self, taskfile):
        self.taskfile = taskfile
        try:
            with open(taskfile, "r") as f:
                self.tasks = json.load(f) 
                self.task_ids = self.tasks["tasks"].keys()
        except IOError:
            logger.error("Unable to load this tasks file: "+taskfile)
    
    def get_task_by_id(self, task_id):
        """
        """
        task_id = unicode(task_id)
        if task_id not in self.task_ids:
            logger.warning("Not a valid task ID in taskfile: "+self.taskfile) 
            return None  # to make the return value explicit
        else:
            return self.tasks["tasks"][task_id]

    def get_random_task(self):
        """
        """
        Settings.set_seed(None)
        self.random_task_id = Settings.random.choice(self.task_ids)
        return self.get_task_by_id(task_id=self.random_task_id)


if __name__=="__main__":
    reload(sys)  
    sys.setdefaultencoding('utf8')  # good practise i think to ensure taskfile is readable on Camdial.org 
    if len(sys.argv) == 1:
        Usage = "python Tasks.py -C config -s random seed" 
        exit(Usage)
    
    parser = argparse.ArgumentParser(description='Task generator')
    parser.add_argument('-C','-c', '--config', help='set config file', required=True, type=argparse.FileType('r'))
    parser.add_argument('-s', '--seed', help='set random seed - overrides configs seed', type=int)
    args = parser.parse_args()

    Settings.init(config_file=args.config.name, seed=args.seed)
    Ontology.init_global_ontology()
    ContextLogger.createLoggingHandlers(config=Settings.config)

    task_generator = TaskCreator()
    task_generator._create()
    task_generator._write()



#END OF FILE
