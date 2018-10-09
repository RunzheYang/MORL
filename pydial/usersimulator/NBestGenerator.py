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
NBestGenerator.py - 
====================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger`

************************

'''
import copy
import os
import numpy as np
from utils import Settings, ContextLogger
logger = ContextLogger.getLogger('')


class NBestGenerator(object):
    def __init__(self, confusion_model, error_rate, nbest_size):
        self.confusionModel = confusion_model
        self.error_rate = error_rate
        self.nbest_size = nbest_size


class EMNBestGenerator(NBestGenerator):
    '''
    Tool for generating random semantic errors.
    '''

    def __init__(self, confusion_model, error_rate, nbest_size):
        super(EMNBestGenerator, self).__init__(confusion_model, error_rate, nbest_size)

    def getNBest(self, a_u):
        '''
        Returns an N-best list of dialogue acts of length nbest_size.
        Each entry is a random confusion of the given dialogue act a_u with probability e (errorRate).
        
        :param a_u: of :class:`DiaActWithProb`
        :type a_u: instance
        '''
        if self.error_rate is None:
            logger.error('Error rate is not specified. Call set_error_rate first.')

        n_best = []
        for i in range(self.nbest_size):
            if Settings.random.rand() < self.error_rate:
                # Creating wrong hyp.
                confused_a_u = self.confusionModel.create_wrong_hyp(a_u)
                confused_a_u.P_Au_O = 1./self.nbest_size
                n_best.append(confused_a_u)
            else:
                a_u_copy = copy.deepcopy(a_u)
                a_u_copy.P_Au_O = 1./self.nbest_size
                n_best.append(a_u_copy)
        return n_best


class EMSampledNBestGenerator(NBestGenerator):
    '''
    The Dirichlet collection NBest generator operates by sampling a length for the N-best list and then
    sampling from a specific Dirichlet for that length.

    This is a derived class from base :class:`NBestGenerator`

    :param confusion_model: of :class:`ConfusionModel`
    :type confusion_model: instance
    :param nbest_size: None
    :type nbest_size: int

    .. note:: The original C++ implementation did not sample N which is the length of the N-best list.
    '''
    def __init__(self, confusion_model, error_rate, nbest_size):
        super(EMSampledNBestGenerator, self).__init__(confusion_model, error_rate, nbest_size)
        self.confidence = [1./nbest_size] * nbest_size

    def getNBest(self, a_u):
        '''
        :param a_u: of :class:`DiaActWithProb`
        :type a_u: instance
        :returns: (list) of such dialogue act types as input
        '''
        if self.error_rate is None:
            logger.error('Error rate is not specified. Call set_error_rate first.')

        n_best = []
        size = Settings.random.choice(range(1, self.nbest_size+1))
        for i in range(size):
            if Settings.random.rand() < self.error_rate:
                new_hyp = self.confusionModel.create_wrong_hyp(a_u)
            else:
                new_hyp = copy.deepcopy(a_u)
            new_hyp.P_Au_O = 1./size
            n_best.append(new_hyp)
        return n_best


class DSTC2NBestGenerator(NBestGenerator):
    '''
    Tool for generating random semantic errors based on the statistics learned from the DSTC2 corpus
    '''
    def __init__(self, confusion_model, error_rate, nbest_size, paramset=None):
        super(DSTC2NBestGenerator, self).__init__(confusion_model, error_rate, nbest_size)

        # The following probabilities are learned from DSTC2 statistics
        self.inc_nb_pos_dist = [0.0,
                                 0.172275641,
                                 0.0580128205,
                                 0.0264423077,
                                 0.0158653846,
                                 0.0110576923,
                                 0.00560897436,
                                 0.0078525641,
                                 0.00288461538,
                                 0.000641025641,
                                 0.699358974]
        self.cor_nb_len_dist = [0.26806104,
                                 0.20744851,
                                 0.13531107,
                                 0.09988262,
                                 0.08291538,
                                 0.06573471,
                                 0.05559705,
                                 0.05783801,
                                 0.02443709,
                                 0.00277452]
        self.inc_nb_len_dist = [0.31794872,
                                 0.06554487,
                                 0.08237179,
                                 0.06570513,
                                 0.07387821,
                                 0.08669872,
                                 0.10721154,
                                 0.12996795,
                                 0.06073718,
                                 0.0099359]

        if paramset:
            if os.path.isfile(paramset):
                with open(paramset, 'r') as paramfile:
                    for line in paramfile:
                        if not line.startswith('#'):
                            if 'incorrectNBPosDist' in line:
                                line = line.split('#')[0].split('=')[1]
                                line = line.replace('[', '').replace(']', '')
                                self.inc_nb_pos_dist = [float(x.strip()) for x in line.split(',')]
                            elif 'incorrectNBLenDist' in line:
                                line = line.split('#')[0].split('=')[1]
                                line = line.replace('[', '').replace(']', '')
                                self.inc_nb_len_dist = [float(x.strip()) for x in line.split(',')]
                            elif 'correctNBLenDist' in line:
                                line = line.split('#')[0].split('=')[1]
                                line = line.replace('[', '').replace(']', '')
                                self.cor_nb_len_dist = [float(x.strip()) for x in line.split(',')]
        else:
            logger.error('Error model config file "{}" does not exist'.format(paramset))
        self.inc_nb_pos_dist = np.array(self.inc_nb_pos_dist)/sum(self.inc_nb_pos_dist)
        self.cor_nb_len_dist = np.array(self.cor_nb_len_dist)/sum(self.cor_nb_len_dist)
        self.inc_nb_len_dist = np.array(self.inc_nb_len_dist)/sum(self.inc_nb_len_dist)





    def getNBest(self, a_u):
        '''
        Returns an N-best list of dialogue acts of length self.nbest_size

        :param a_u: of :class:`DiaActWithProb`
        :type a_u: instance
        '''
        if self.error_rate is None:
            logger.error('Error rate is not specified. Call set_error_rate first.')
        n_best = []

        # decide if its correct or incorrect
        if Settings.random.rand() < self.error_rate:
            # incorrect sample
            # sample nbest size
            nb_size = np.argmax(Settings.random.multinomial(1,self.inc_nb_len_dist)) + 1
            if nb_size > self.nbest_size:
                nb_size = self.nbest_size
            # sample the correct hyp position in the nb list
            corr_pos = np.argmax(Settings.random.multinomial(1,self.inc_nb_pos_dist))

        else:
            # correct sample
            # sample nbest size
            nb_size = np.argmax(Settings.random.multinomial(1, self.cor_nb_len_dist)) + 1
            if nb_size > self.nbest_size:
                nb_size = self.nbest_size
            corr_pos = 0

        # generate the nbest list
        for i in range(nb_size):
            if corr_pos == i:
                a_u_copy = copy.deepcopy(a_u)
                a_u_copy.P_Au_O = 1.
                n_best.append(a_u_copy)
            else:
                confused_a_u = self.confusionModel.create_wrong_hyp(a_u)
                while confused_a_u == a_u or confused_a_u in n_best:
                    confused_a_u = self.confusionModel.create_wrong_hyp(a_u)  # enforce it to be confused
                confused_a_u.P_Au_O = 0.01
                n_best.append(confused_a_u)

        return n_best

#END OF FILE