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
ConfidenceScorer.py -
===================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.DiaAct` |.|


************************

''' 

__author__ = "cued_dialogue_systems_group"

import numpy as np
import os
from utils import Settings, DiaAct, ContextLogger
logger = ContextLogger.getLogger('')

class AdditiveConfidenceScorer(object):
    '''Additive confidence scoring of TODO
    '''

    def __init__(self, topProb1=False, rescale=False):
        self.rescale = rescale
        self.TOP_PROB_IS_ONE = topProb1

    def assignConfScores(self, dapl):
        '''
        :param dapl: N-best list of DiaAct
        :type dapl: list
        :returns: scored dialogue act list
        '''
        outlist = []
        outdactlist = []
        total = 0.0
        for hyp in dapl:  # for each hyp in the given N-best list
            total += hyp.P_Au_O
            if hyp not in outdactlist:  # Add this hyp
                outlist.append(hyp)
                outdactlist.append(hyp)
            else:  # or add P_Au_O
                i = outdactlist.index(hyp)
                outlist[i].P_Au_O += hyp.P_Au_O

        # Rescale
        if total > 1 or self.rescale:
            for h in outdactlist:
                h.P_Au_O = float(h.P_Au_O) / total

        outlist.sort()
        if self.TOP_PROB_IS_ONE and outlist:
            outlist = [outlist[0]]
            outlist[0].P_Au_O = 1.0

        return outlist

class DSTC2ConfidenceScorer(object):
    '''Confidence scorer based on the statistics obtained from the DSTC2 corpus
    '''

    def __init__(self, paramset=None):

        #these are the original dstc2 statistics
        # statistics for correct items
        cor_u0 = 0.869149127189
        cor_u1 = 0.103655660074
        cor_u2 = 0.0457239494195
        cor_u3 = 0.0327489814707
        cor_u4 = 0.0271721415285
        cor_u_rest = 0.0503603589269
        self.cor_u = [cor_u0, cor_u1, cor_u2, cor_u3, cor_u4, cor_u_rest]

        cor_var0 = 0.0290966884125
        cor_var1 = 0.0103859010664
        cor_var2 = 0.0017143689477
        cor_var3 = 0.000545328421009
        cor_var4 = 0.000276389806598
        cor_var_rest = 0.00141147536408
        cor_var = [cor_var0, cor_var1, cor_var2, cor_var3, cor_var4, cor_var_rest]
        self.cor_std = [np.sqrt(x) for x in cor_var]

        # statistics for incorrect items
        inc_u0 = 0.762642732156
        inc_u1 = 0.176619385495
        inc_u2 = 0.0767276824481
        inc_u3 = 0.0486142019601
        inc_u4 = 0.0373167405546
        inc_u_rest = 0.0662623414006
        self.inc_u = [inc_u0, inc_u1, inc_u2, inc_u3, inc_u4, inc_u_rest]

        inc_var0 = 0.0517617320729
        inc_var1 = 0.0133762671331
        inc_var2 = 0.00275666881632
        inc_var3 = 0.000807049241929
        inc_var4 = 0.000359077683994
        inc_var_rest = 0.00183093114343
        inc_var = [inc_var0, inc_var1, inc_var2, inc_var3, inc_var4, inc_var_rest]
        self.inc_std = [np.sqrt(x) for x in inc_var]

        if paramset:
            if os.path.isfile(paramset):
                with open(paramset, 'r') as paramfile:
                    for line in paramfile:
                        if not line.startswith('#'):
                            if 'incorrectMean' in line:
                                line = line.split('#')[0].split('=')[1]
                                line = line.replace('[', '').replace(']', '')
                                self.inc_u = [float(x.strip()) for x in line.split(',')]
                            elif 'correctMean' in line:
                                line = line.split('#')[0].split('=')[1]
                                line = line.replace('[', '').replace(']', '')
                                self.cor_u = [float(x.strip()) for x in line.split(',')]
                            elif 'incorrectVar' in line:
                                line = line.split('#')[0].split('=')[1]
                                line = line.replace('[', '').replace(']', '')
                                self.inc_std = [np.sqrt(float(x.strip())) for x in line.split(',')]
                            elif 'correctVar' in line:
                                line = line.split('#')[0].split('=')[1]
                                line = line.replace('[', '').replace(']', '')
                                self.cor_std = [np.sqrt(float(x.strip())) for x in line.split(',')]
            else:
                logger.error('Error model config file "{}" does not exist'.format(paramset))

    def assignConfScores(self, dapl):
        '''
        :param dapl: N-best list of DiaAct
        :type dapl: list
        :returns: scored dialogue act list
        '''

        if np.argmax([x.P_Au_O for x in dapl]) == 0: # check if the generated nbest corresponds to a correct or incorrect sample
            u = self.cor_u
            std = self.cor_std
        else:
            u = self.inc_u
            std = self.inc_std

        # sample each confidence score in the nbest list
        for i in range(len(dapl)):
            sample = Settings.random.normal(u[i], max(0.0001,std[i]))
            sample = round(max(min(0.99, sample), 0.01), 4)
            dapl[i].P_Au_O = sample

        # sample the 'rest' confidence score
        rest_cs = 0
        for i in range(len(dapl), len(u)):
            sample = Settings.random.normal(u[i], max(0.00001, std[i]))
            sample = round(max(min(0.99, sample), 0.01), 4)
            rest_cs += sample
        nb_names = [x.to_string() for x in dapl]
        if 'null()' in nb_names:
            dapl[nb_names.index('null()')].P_Au_O += rest_cs
        else:
            rest_da = DiaAct.DiaActWithProb('null()')
            rest_da.P_Au_O = rest_cs
            dapl.append(rest_da)

        # Rescale
        total = np.sum([x.P_Au_O for x in dapl])
        for h in dapl:
            h.P_Au_O = float(h.P_Au_O) / total

        '''# check nbest list correctness
        assert abs(sum([x.P_Au_O for x in dapl]))-1 < 0.0001
        if len(nb_names) > len(set(nb_names)):
            print nb_names
            import sys
            sys.exit()
        for h in dapl:
            cs = h.P_Au_O
            assert cs <= 1
            assert cs >= 0'''

        return dapl


#END OF FILE
