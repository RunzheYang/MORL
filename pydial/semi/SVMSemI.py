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
SVMSemI - Support Vector Machine Semantic Decoder
=====================================================================

To use this in pydial, need to set "semitype = SVMSemI" for a domain in the relevant interface config file
(in the current state it is the CamRestaurants domain)
See texthub_svm.cfg, which can be used for this purpose for texthub interface

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: Discriminative Spoken Language Understanding Using Word Confusion Networks
    http://mi.eng.cam.ac.uk/~sjy/papers/hgtt12.pdf

.. seealso:: CUED Imports/Dependencies:

    import :mod:`semi.SemI` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod: `decode.svmdec` |semi/CNetTrain|


************************

Important: please see semi/CNetTrain/README.md

'''


import os, sys, ConfigParser
from utils import ContextLogger
logger = ContextLogger.getLogger('')
import imp
from semi import SemI

old_path = os.getcwd()
if "semi" not in old_path:
    path = old_path+"/semi/CNetTrain/"
else:
    path = old_path+"/CNetTrain/"
os.sys.path.insert(1, path)
import decode as svmdec
import math
import time
import RegexSemI
print sys.path

__author__ = "cued_dialogue_systems_group"

class SVMSemI(SemI.SemI):

    def __init__(self):
        '''
        Initialise some objects, use RegexSemI to solve classification errors, and to deal with
        googdbye and request for alternatives
        :return:
        '''

        self.RSemI = RegexSemI.RegexSemI() # For goodbye and request alternatives in decode
        self.config = ConfigParser.ConfigParser()
        self.config.read(path+"/config/eg.cfg")
        self.classifiers=svmdec.init_classifier(self.config)
        self.sys_act = []


    def decode(self, ASR_obs, sys_act=None, turn=None):
        '''
        Includes os.chdir to change directories from pydial root to the locally installed inside semi directory.
        Directories are changed back to pydial root after prediction. This ensures all the required
        config and data files are accessed.
        :param ASR_obs: hypothesis with the ASR N-best list
        :param sys_act: previous system dialogue act
        :param turn:  turn id
        :return: Semantic representation from the asr output
        '''

        #Check first general dialogue acts with Regular Expressions
        regexpred = self.decode_general_hypothesis(ASR_obs[0][0])

        if "bye()" in regexpred:
            return [("bye()", 1.0)]
        elif "reqalts()" in regexpred:
            return [("reqalts()", 1.0)]
        elif "affirm()" in regexpred:
            return [("affirm()",1.0)]
        elif "negate()"in regexpred:
            return [("negate()",1.0)]
        elif "hello()" in regexpred:
            return [("hello()",1.0)]
        else:
            old_path = os.getcwd()
            os.chdir(path)

            sentinfo = self.input_json(ASR_obs, self.sys_act, turn)

            before = int(round(time.time() * 1000))
            predictions = svmdec.decode(self.classifiers,self.config, sentinfo)

            after = int(round(time.time() * 1000))
            pred_dur = after - before
            logger.debug("prediction time: %d" % pred_dur) # Time taken by DLSemI for prediction

            os.chdir(old_path)

            logger.info(predictions)
            self.semActs = self.format_semi_output(predictions)

            logger.info(self.semActs)
            return self.semActs


    def input_json(self, ASR_obs, sys_act, turn):
        '''
        Format the incoming ASR_obs and sys_act into an input for SVM Classifiers in JSON
        :param ASR_obs: ASR hypothesis
        :param sys_act: Last system action
        :param turn:  Turn id
        :return:
        '''

        logger.info(ASR_obs)

        sentinfo = {}

        asrhyps = []
        for obs in ASR_obs:
            asrhyps.append(dict([ (u'asr-hyp', unicode(obs[0])), (u'score', math.log(obs[1]))]))

        sentinfo['turn-id'] = turn
        sentinfo['asr-hyps'] = asrhyps
        sentinfo['prevsysacts'] = []

        return sentinfo


    def format_semi_output(self, sluhyps):
        '''
        Transform the output of SVM classifier  to make it compatible with cued-pydial system
        :param sluhyps: output coming from SVMSemI
        :return: SVMSemI output in the required format for cued-pydial
        '''

        prediction_clean=[]
        for hyp in sluhyps:
            if not hyp["slu-hyp"]:
                prediction_clean = [('null()',hyp['score'])]
                continue


            probability = hyp['score']
            slu_hyp=hyp["slu-hyp"]

            for sluh in slu_hyp:
                dact = sluh['act']
                pred_str=unicode(dact)
                prediction_string = []
                if not sluh['slots']:
                    prediction_string.append(pred_str+"()")


                for slot in sluh['slots']:
                    prediction_string.append('%s(%s=%s)' % (unicode(dact), unicode(slot[0]), unicode(slot[1])))


                prediction_string = '|'.join(prediction_string)

                prediction_clean.append((prediction_string, probability))



        return prediction_clean


    def decode_general_hypothesis(self, obs):
        '''
        Regular expressions for bye() and reqalts(), affirm and type
        :param obs: ASR hypothesis
        :return: RegexSemI recognised dialogue act
        '''

        self.RSemI.semanticActs = []

        self.RSemI._decode_reqalts(obs)
        self.RSemI._decode_bye(obs)
        self.RSemI._decode_type(obs)
        self.RSemI._decode_affirm(obs)

        return self.RSemI.semanticActs


if __name__ == '__main__':
    svm=SVMSemI()
    #preds=svm.decode([('I am looking for a chinese restaurant in the north',1.0)])
    preds=svm.decode([('I am looking for restaurant',1.0)])
    print preds
    preds=[]
    #preds=svm.decode([('something in the north',1.0)])
    preds=svm.decode( [(' I am looking for a cheap restaurant', 1.0)])
    print preds
    preds=svm.decode( [('something in the north', 1.0)])
    print preds