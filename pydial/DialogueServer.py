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
DialogueServer.py - Audio and VoIP interface to Spoken SDS 
============================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

This class implements an HTTP server, it receives HTTP Requests and generates HTTP Replies.
The format of the messages is JSON. The server has an agent (i.e., a dialogue system or the set of dialogue system components),
that can be copied, and can take multiple calls in this way
it can correctly handle concurrent request already supported by the BaseHTTPServer


**Basic Execution**: 
    >>> python DialogueServer.py -c CONFIG [--nocolor]

   
**Important Config variables** [Default values]::

    [dialogueserver]
    dialhost = localhost 
    dialport = 8082

*Please change these configuration variables according to your machine settings.*
*Note that these variables must agree with the configuration of your HTTP speech client*
*If you want to run it locally use: localhost*

Protocol description
--------------------
**Requests to the server**

New call: notify the DialogueManager when a new call (or new session) has started::

    newcall?sessionpar={"session": "SESSION_NAME"}
    
Next: ask the DialogueManager what to do next and provide the JSON RESULT from the ASR or the DTMF RESULT::

    next?{ "session": "SESSION_NAME", "result" : "JSON_ASR_RESULT"}
    next?{ "session": "SESSION_NAME", "result" :{"dtmf" :  "DTMF_RESULT" }"
    

Clean: clean session in the case of unexpected errors in the ASR CLIENT or forced hung-up in the VOICE CLIENT::

    clean?{"session": "SESSION_NAME"}
    
    
**Responses from the server**

Question::

    http reply {"bargein": "true", "replyType": "question",
                "text": "Hello, welcome to the Cambridge Multi-Domain dialogue system. How may I help you?"}
    
Prompt::

    http reply {"bargein": "false", "replyType": "prompt",
                "text": "Thank you, Goodbye", "final":"true"}

Control::

    http reply {"return_control": "true", "replyType": "control",
                "session_kept_alive":"true"}
    
Flags for the messages are::

    "replyType" = "prompt"|"question""|"control"
    "bargein"   = "true"|"false", is barge in supported in the ASR?
    "final"     = "true"|"false", if this flag is true the Voice Client must hung up the call.
    "dtmf"      = "true"|"false", is expecting dtmf result?
    "dtmfsize"  =  dtmfsize (i.e., number of digits)
    "text"      =  TEXT_TO_PROMPT
    "return_control" = "true"|"false", indicates ood signal from client
    "session_kept_alive" = "true"|"false", ood signal received but server keeping going for now


.. seealso:: CorneliaServer:
    https://bitbucket.org/lmr46/voiceserver.git
    VoiceBroker (The Class Server):
    https://bitbucket.org/dialoguesystems/voicebroker.git

    import :`json` |.|
    import :BaseHTTPServer |.|
    import :json |.|
    import :Settings |.|
    import :ContexLogger |.|

************************
 
'''
import argparse
import json
import BaseHTTPServer

import Agent
from utils import Settings
from utils import ContextLogger
from ontology import Ontology
logger = ContextLogger.getLogger('root')

__author__ = "cued_dialogue_systems_group"
__version__ = Settings.__version__


#================================================================================================
# SERVER BEHAVIOUR
#================================================================================================
def make_request_handler_class(dialServer):
    """
    """
    class RequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
        '''
            Process HTTP Requests
            :return:
        '''

        def do_POST(self):
            '''
            Handle only POST requests. Please note that GET requests ARE NOT SUPPORTED!
            '''
            self.error_free = True # boolean which can become False if we encounter a problem
            agent_id = None
            self.currentSession=None
            prompt_str=''
            reply={}
            
            consent_str = 'This call will be recorded for research purposes. Your continued participation serves as express consent to be recorded. '
            
            #------Get the "Request" link.
            print '-'*30
            request = self.path[1:] if self.path.find('?') < 0 else self.path[1:self.path.find('?')]

            logger.debug('Request: ' + str(request))
            logger.debug('POST full path: %s ' % self.path)
            if not 'Content-Length' in self.headers:
                data_string=self.path[self.path.find('?')+1:]
            else:
                data_string = self.rfile.read(int(self.headers['Content-Length']))
            
            logger.debug("Request Data:"+data_string)  # contains e.g:  {"session": "voip-5595158237"}
            
            recognition_fail = True # default until we confirm we have received data
            try:
                data = json.loads(data_string)     # ValueError
                self.currentSession = data["session"]   # KeyError
            except Exception as e:
                logger.warning("Not a valid JSON object (or object lacking info) received. %s" % e)
            else:
                recognition_fail = False

            #----- look for context information coming in (as per DialPort usage):
            info = dialServer.parse_additional_information(data)
            start_domain = info['start_domain']
            
            #----- first do handling of return control requests
            if info['return_control']:
                logString = "Handing back control to server."
                logger.info(logString)
                reply = dialServer.control(True, self.currentSession, info['session_kept_alive'])
                prompt_str = "Handling terminate request"
                if not info['session_kept_alive']:
                    logString = " Session is not kept alive."
                    agent_id = dialServer.agent_factory.end_call(session_id=self.currentSession)
                else:
                    logString = " Session is kept alive." 
                logger.info(logString)
                

            #------Work through cases: 1-fail, 2-new, 3-continue, 4-clean
            elif recognition_fail:
                self.error_free = False
                reply,prompt_str = dialServer.recognition_failed_response(msg='MISSING_INFO', session_id=self.currentSession)
            elif request=='newcall':
                # A new conversation ( or call in the case of VoIP) has started
                
                try:
                    sys_act, agent_id = dialServer.agent_factory.start_call(session_id=self.currentSession,
                                                                               start_domain=start_domain)
                    prompt_str = sys_act.prompt
                except:
                    self.error_free = False
                    logger.warning("Tried to start a new call with a session id: {} already in use".format(self.currentSession))
                    reply,prompt_str = dialServer.recognition_failed_response(msg='SESSION_ID_IN_USE', session_id=self.currentSession)
                else:
                    logger.info("A new call has started. Session: %s " % self.currentSession)
                    if dialServer.issueConsent:
                        prompt_str = consent_str + prompt_str
                    reply = dialServer.question(prompt_str, session_id=self.currentSession, isbargein=True)
                
                
                #------------------------------------------------------------------------------------------------

            elif request=='next':
                # Next step in the conversation flow 

                # map session_id to agent_id
                try:
                    agent_id = dialServer.agent_factory.retrieve_agent(session_id=self.currentSession)
                except: # Throws a ExceptionRaisedByLogger
                    self.error_free = False
                    logger.warning("Tried to get an agent for the non-existent session id: {}".format(self.currentSession))
                    reply,prompt_str = dialServer.recognition_failed_response(msg='NO_SUCH_SESSION', session_id=self.currentSession)
                else:
                    logger.info("Continuing session: %s with agent_id %s " % (self.currentSession, agent_id))
                 

                if self.error_free and "result" in data and not dialServer.agent_factory.query_ENDING_DIALOG(agent_id):
                    if data["result"] =="ok":
                        # This means the ASR was only ordered to Prompt a text (to Synthesise),
                        # thus there is not recognition result at all. Therefore, it does not need to do anything in here ...
                        # We are here whenever the system send prompts to the Broker.
                        # For instance, the sentence sends a welcome affirmation (not recognition is received) to the broker and
                        # then asks what to do (thus a recognition is received),
                        # that will depend on the design of the dialogue and must be supported - lina.
                        logger.info("ASR: None (ie voicebroker issued prompt only" )
                        reply,prompt_str = dialServer.recognition_failed_response(msg='no data',session_id=self.currentSession)
                    else:
                        #Here the VoiceBroker did send a recognition result
                        asr_info=dialServer.cleaningCNet(data)
                        logger.info("ASR: "+ str(asr_info))
                        sys_act = dialServer.agent_factory.continue_call(agent_id, asr_info, domainString=start_domain)
                        prompt_str = sys_act.prompt
                        
                        # Are we ENDING the dialog here? 
                        if not dialServer.agent_factory.query_ENDING_DIALOG(agent_id):
                            reply=dialServer.question(prompt_str, session_id=self.currentSession, isbargein=True)
                    

            elif request=='clean':
                #receives clean request  either because of user hung-up, Ctrl+C, or any other exception ...
                logger.info("Received request to Clean Session ID from the VoiceBroker...:"+self.currentSession)
                self.error_free = False                
                try:
                    agent_id = dialServer.agent_factory.end_call(session_id=self.currentSession,noTraining=True)
                    reply,prompt_str = dialServer.cleaning_response(msg='CLEANING-success', session_id=self.currentSession)
                except: # an ExceptionRaisedByLogger
                    logger.warning("Tried to get an agent for the non-existent session id: {}".format(self.currentSession))
                    reply,prompt_str = dialServer.cleaning_response(msg='CLEANING-failed-no-such-session', 
                                                                    session_id=self.currentSession)


            #------ENDING THE DIALOGUE: GET TASK ID AND TASK -- even if we eval by RNN we still want this
            if self.error_free and dialServer.agent_factory.query_ENDING_DIALOG(agent_id):
                logger.info("Agent %s is ending dialogue",agent_id)
                # A. Try to get the task ID - either via DTMF or keyboard ----------------------------------
                if dialServer.COLLECT_TASK and dialServer.agent_factory.agents[agent_id].task is None:
                    if dialServer.agent_factory.agents[agent_id].TASK_RETRIEVAL_ATTEMPTS == 0:
                        prompt_str,reply = dialServer.ask_for_task_id(agent_id, currentPrompt=prompt_str)
                    else:
                        # See if the DTMF number received points to a valid task:
                        #if task_id is -1 -> will get None
                        task_id = dialServer.get_task_id_DTMF(agent_id, data)
                        logger.info("task_id %s" %task_id)
                        dialServer.agent_factory.agents[agent_id].task = dialServer.get_task_by_id(task_id=task_id)
                        dialServer.agent_factory.agents[agent_id].taskId = task_id
                    if dialServer.agent_factory.agents[agent_id].TASK_RETRIEVAL_ATTEMPTS > 1:  # 1 : just ask for msg
                        if dialServer.agent_factory.agents[agent_id].task is None:
                            # retry to get a valid task_id
                            prompt_str = "Please try to enter task number again followed by the hash key. You can try to add small pauses between each key press."
                            reply=dialServer.prompt(prompt_str,
                                                    session_id=self.currentSession,
                                                    isbargein=False, isfinal=False, isdtmf=True, dtmfsize=5)
                        else:
                            logger.info("Successfully obtained a task")
                            prompt_str = dialServer.RECEIVED_DTMF_MSG  # got it for task_id --

                # B. Get subjective feedback ------------------------------------------------------------
                if dialServer.COLLECT_SUBJECTIVE_FEEDBACK and \
                        (dialServer.agent_factory.agents[agent_id].subjective is None or dialServer.agent_factory.agents[agent_id].subjective2 is None) and \
                        ((dialServer.COLLECT_TASK and dialServer.agent_factory.agents[agent_id].task is not None) \
                        or dialServer.COLLECT_TASK is False):
                    if dialServer.agent_factory.agents[agent_id].subjective is None:
                        if dialServer.agent_factory.agents[agent_id].SUBJECTIVE_RETRIEVAL_ATTEMPS == 0:
                            prompt_str, reply = dialServer.ask_for_subjective(agent_id, currentPrompt=prompt_str)
                        else:
                            dialServer.agent_factory.agents[agent_id].subjective = dialServer.get_subjective_DTMF(agent_id,data,drange=[0,1])
                            logger.info("Subjective feedback "+str(dialServer.agent_factory.agents[agent_id].subjective))
                            
                            if dialServer.agent_factory.agents[agent_id].subjective is None:
                                #retry to get valid subjective feedback (0or1) The hash key is only for more than 1 digit
                                prompt_str = "Please try again."
                                prompt_str, reply = dialServer.ask_for_subjective(agent_id, currentPrompt=prompt_str)
                            else:
                                prompt_str = dialServer.RECEIVED_DTMF_MSG  # got it for subjective 0/1 --
                                
                # C. Complete the dialog --------------------------------------------------------------------
                FEEDBACK_LOGIC = dialServer.agent_factory.agents[agent_id].subjective is not None or \
                        dialServer.COLLECT_SUBJECTIVE_FEEDBACK is False         # got it OR didnt want it
                TASK_LOGIC = dialServer.agent_factory.agents[agent_id].task is not None or \
                        dialServer.COLLECT_TASK is False
                if TASK_LOGIC and FEEDBACK_LOGIC:
                    if 'local-audio' not in dialServer.agent_factory.agents[agent_id].session_id and dialServer.GENERATE_TOKEN:
                        # Post landing page validation token to Camdial server
                        token_text,token = dialServer.generate_token()
                        logger.info("Token for camdial feedback form: "+str(token))
                        prompt_str += " "+token_text
                        dialServer.postToCamdial(token=token, dialogueID=self.currentSession, task=dialServer.agent_factory.agents[agent_id].taskId)
                    else:
                        if prompt_str is None:
                            prompt_str = ''
                        prompt_str += 'You can now hang up.'
                    reply=dialServer.prompt(prompt_str,session_id=self.currentSession, isbargein=False,isfinal=True)
                    # Important! Now finish the learning etc for the just finished dialog
                    dialServer.agent_factory.end_call(agent_id=agent_id)
                
            #------ Completed turn --------------
            
            # POST THE REPLY BACK TO THE SPEECH SYSTEM
            logger.info("Sending prompt: "+prompt_str+" to tts.")
            self.send_response(200)  # 200=OK W3C HTTP Standard codes
            self.send_header('Content-type', 'text/json')
            self.end_headers()
            self.wfile.write(reply)
            
    return RequestHandler



#================================================================================================
# DIALOGUE SERVER
#================================================================================================
class DialogueServer(object):
    '''
     This class implements an HTTP Server
    '''
    def __init__(self):
        """    HTTP Server
        """
        self.RECEIVED_DTMF_MSG =  "Got it, thanks."
        self.GENERATE_TOKEN = True 
        self.COLLECT_SUBJECTIVE_FEEDBACK = False
        self.COLLECT_TASK = False
        self.host="localhost"
        self.port=8082
        self.TOKENSERVERURL = "http://www.camdial.org/~djv27/mt-multiDomain/receive-token.py" 
        # DTMF prompt given over voip call at dialogue's end
        self.COLLECT_TASKID_PROMPT = ".  Please now enter the 5 digit task number followed by the hash key"
        #The hash key is only for multiple digits
        self.COLLECT_SUBJECTIVE_PROMPT = " Have you found all the information you were looking for? Please enter one for yes, and zero for no."
        
        self.ood_count = 0
        self.OOD_THRESHOLD = 1
        
        # Speech Settings:
        self.allowbargein = False  # TODO - config this or remove it -- OR SHOULD IT BE SOMETHING SENT FROM VOICEBROKER EACH TURN?
        if Settings.config.has_option("dialogueserver","tokenserverurl"):
            self.TOKENSERVERURL = Settings.config.get("dialogueserver","tokenserverurl")
            self.TOKENSERVERURL = self.TOKENSERVERURL.strip('"')
        if Settings.config.has_option("dialogueserver","generatetoken"):
            self.GENERATE_TOKEN = Settings.config.getboolean("dialogueserver","generatetoken")
        if Settings.config.has_option("dialogueserver","collecttask"):
            self.COLLECT_TASK = Settings.config.getboolean("dialogueserver","collecttask")
        if Settings.config.has_option("dialogueserver","subjectivefeedback"):
            self.COLLECT_SUBJECTIVE_FEEDBACK = Settings.config.getboolean("dialogueserver","subjectivefeedback")
        if Settings.config.has_option("dialogueserver","subjectivefeedbackprompt"):
            self.COLLECT_SUBJECTIVE_PROMPT = Settings.config.getboolean("dialogueserver","subjectivefeedbackprompt")
        if Settings.config.has_option("dialogueserver","dialhost"):
            self.host = Settings.config.get("dialogueserver","dialhost")
            self.host = self.host.strip('"')
        if Settings.config.has_option("dialogueserver","dialport"):
            self.port = Settings.config.getint("dialogueserver","dialport")
            
        self.issueConsent = False
        if Settings.config.has_option("dialogueserver","issueConsent"):
            self.issueConsent = Settings.config.getboolean("dialogueserver","issueConsent")
                
        
        # Dialogue agent:
        self.agent_factory = Agent.AgentFactory(hub_id='dialogueserver')
        
        self.tasks = None 
        if Settings.config.has_option("dialogueserver","tasksfile"):
            from tasks import Tasks
            self.tasks = Tasks.TaskReader(taskfile = Settings.config.get("dialogueserver","tasksfile")) 
        if self.COLLECT_TASK and self.tasks is None:
            logger.error("Must provide a tasks file if you want to collect task")


    def run(self):
        """Listen to request in host dialhost and port dialport""" 
        RequestHandlerClass = make_request_handler_class(self)
        server = BaseHTTPServer.HTTPServer((self.host, self.port), RequestHandlerClass)
        logger.info('Server starting %s:%s (level=%s)' % (self.host, self.port, 'info'))
        try:
            while 1:
                server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            logger.info('Server stopping %s:%s' % (self.host, self.port))
            server.server_close()
            # prints, complete learning and policy saves:
            self.agent_factory.power_down_factory()
    
    def parse_additional_information(self, data):
        '''The ASR client communicating with DialogueServer may also send some additional information. This is included under the
        "context" key of the JSON object it posts, and can be extended to included any extra information. For now it just contains
        the domain control should start with, since with DialPort for example they do their own topic tracking and hand control off
        to our system with the desire to access just a sub domain (eg SFRestaurants) and not start with our topicmanager.

        :return: dict --  keys: 'start_domain' = initial domain hint from client;
                                'return_control' = True,  return control to client;
                                'session_kept_alive' = True, client is requesting return of control but currently resisting.
        
        .. note:: the format for this additional information, within the original JSON object being received is as follows:
                  
                  >>> "context": {
                  >>>       "init_domain": [
                  >>>           { "value": "SFRestaurants", "confidence": 0.6},
                  >>>           { "value": "SFHotels", "confidence": 0.1}
                  >>>       ],
                  >>>       "entities": [
                  >>>           {"type": "LOC", "value": "pittsburgh"}
                  >>>       ]
                  >>>  },
                  >>>  "ood":"another_domain",
                  >>>  "sudoTerminal":"True"

                  where
                    - "init_domain" is an N-best list of possible domains identified by the voicebroker
                    - "entities" provides relevant context information such as the geo location "LOC":
                    - "ood" is used to indicate that the voicebroker thinks that the domain has changed.
                      If "ood" is repeated more than OOD_THRESHOLD times, then control is returned
                      if "ood" is null, then ood counter is reset.
                    - "sudoTerminal" is true, then server returns control immediately.
        
        '''
        info = dict()
        info['start_domain'] = None
        info['return_control'] = None
        info['session_kept_alive'] = None
        
        if "context" in data:
            start_domain = None
            logger.info("[DialPort]: context information received in JSON data")
            try:
                # not good style ... but just going to wrap several lines in a try here and catch general exception to handle 
                # any changes CMU might make to format of JSON msgs being posted to us.
                topictracker_hyps = data["context"]["init_domain"]
                _prob = 0
                for dom_prob in topictracker_hyps:
                    if dom_prob["confidence"] > _prob:
                        start_domain = dom_prob["value"]
                        _prob = dom_prob["confidence"]
            except KeyError as e:
                logger.warning('[DialPort]: Problem parsing dialport messages - JSON is missing the key: {}'.format(e))
            except Exception as e:
                logger.warning('[DialPort]: Problem parsing dialport messages {}'.format(e))
            
            # Ensure start_domain is valid
            if start_domain not in Ontology.global_ontology.ontologyManagers:  
                # includes wikipedia too, which is an illegal start domain. NB: accessing domains list through global_ontology rather
                # than importing OntologyUtils which has available_domains.
                logger.warning('The received start domain: {} is not a valid domain'.format(start_domain))
                start_domain = None
            else:
                logger.info('External instruction to dialogue server to hand control to domain {}'.format(start_domain))
            info['start_domain'] = start_domain
        if "ood" in data:
            try:    
                if data["ood"]: # OOD detected
                    logger.debug("[DialPort]: ood detected")
                    info['session_kept_alive'] = True
                    self.ood_count += 1 # simple heuristic: if ood occurs subsequently often enough, return control
                    info['return_control'] = self.ood_count > self.OOD_THRESHOLD
                else:
                    self.ood_count = 0
                    info['return_control'] = False
            except KeyError as e:
                logger.warning('[DialPort]: Problem parsing dialport messages - JSON is missing the key: {}'.format(e))
            except Exception as e:
                logger.warning('[DialPort]: Problem parsing dialport messages {}'.format(e))
        else:
            self.ood_count = 0
        if "sudoTerminal" in data:
            if data["sudoTerminal"]:
                info['return_control'] = True
                info['session_kept_alive'] = False
        return info

    def ask_for_task_id(self,agent_id,currentPrompt):
        '''nb - statefull 
        '''
        self.agent_factory.agents[agent_id].TASK_RETRIEVAL_ATTEMPTS += 1   
        prompt_str = currentPrompt + self.COLLECT_TASKID_PROMPT
        session_id = self.agent_factory.agent2session(agent_id)
        reply=self.prompt(prompt_str,session_id,isbargein=False, isfinal=False,isdtmf=True, dtmfsize=5)
        return prompt_str, reply

    def ask_for_subjective(self,agent_id, currentPrompt):
        '''nb - statefull 
        '''
        self.agent_factory.agents[agent_id].SUBJECTIVE_RETRIEVAL_ATTEMPS += 1  
        prompt_str = currentPrompt + " " + self.COLLECT_SUBJECTIVE_PROMPT  # now ask for 0/1
        session_id = self.agent_factory.agent2session(agent_id)
        reply=self.prompt(prompt_str, session_id, isbargein=False,isfinal=False,isdtmf=True, dtmfsize=1)
        return prompt_str, reply

    def _get_DTMF(self,data):
        """For either task id or subjective feedback
        """
        result= data["result"]
        if "dtmf" not in result: 
            logger.warning("No dtmf entry from user/voiceserver") 
            return -1

        dtmf_input = result["dtmf"]
        logger.info("DialogueServer received DTMF : %s " % dtmf_input)
        return dtmf_input

    def get_subjective_DTMF(self, agent_id, data, drange = None):
        """nb - statefull
        """
        #self.agent_factory.agents[agent_id].SUBJECTIVE_RETRIEVAL_ATTEMPS += 1
        dtmf_input = self._get_DTMF(data)
        if dtmf_input == -1:
            return -1
        try:
            int_dtmf = int(dtmf_input)
            if drange is not None:
                if int_dtmf in drange:
                    return int_dtmf
                else:
                    logger.warning("Got a valid number - but not in data range {}.".format(drange))
                    return None
            else:
                return int_dtmf
        except ValueError:
            logger.warning("DTMF Subjective error: non numeric")
        except:
            logger.warning("DTMF Subjective error")
        return None  # failed if we got here

    def get_task_id_DTMF(self, agent_id, data):
        '''
        Note the taskID * 3 = task Number (and it is the task number that is displayed on the landing page, and that will 
        thus be entered here). Note - is statefull  (ie increases TASK_RETRIEVAL_ATTEMPTS)
        ''' 
        self.agent_factory.agents[agent_id].TASK_RETRIEVAL_ATTEMPTS += 1
        dtmf_input = self._get_DTMF(data)
        if dtmf_input == -1:
            return -1
        try:
            int_dtmf = int(dtmf_input)
            if int_dtmf%3 == 0:
                return int_dtmf/3 
        except ValueError:
            logger.warning("You entered non numeric characters")  
        except KeyboardInterrupt:
            raise  # if the user is just pressing ctrl-c - let them quit!
        except:
            logger.warning("Something else went wrong - trying to get DTMF again") 
        return -1  # if we got here, we failed


    def generate_token(self):
        """ Produces the 4 digit token for verification of the call on camdial.
        """
        token = Settings.random.random_integers(low=1000,high=9999)  # 4 digit number  
        token_str = "  ".join(str(token))
        token_text = "The 4 digit token for the feedback form is: "+token_str +" , "
        token_toRepeat = " I repeat: "+token_str+ " ,"
        token_toRepeat *= 3  # repeat msg in case they missed the code
        token_end = " You can now hang up. "
        token_text = token_text+token_toRepeat+token_end
        return (token_text,token)

    def postToCamdial(self,token,dialogueID,task):
        """
        """
        param = self.TOKENSERVERURL + "?token="+str(token)+"&dialogueID="+dialogueID+"&task="+str(task)
        logger.info("MSG for Camdial: "+param)
        import requests  # sudo pip install requests. Or use the requirements file
        info = {"token":str(token), "dialogueID":dialogueID, "task":str(task)}
        try:
            r = requests.get(self.TOKENSERVERURL, params=info)
            if "[Camdial] SAVED Token" not in r.text:   # this depends on the receive-token.py file camdial side
                logger.warning("Camdial communication error")
            else:
                logger.info("Success. Received following feedback from Camdial: "+r.text)
        except Exception as e:
            logger.warning("An error occured in communication with Camdial: "+str(e))

    def get_task_by_id(self,task_id):
        """Gets a task by task_id. returns None if no such task_id exists 
        """ 
        return self.tasks.get_task_by_id(task_id = task_id)

    def cleaning_response(self, msg='', session_id=None, isbargein=None):
        '''Note that msg still states: DIALOGUE_SERVER_ERROR - since session should only need to be cleaned if user hung up etc - in
        normal flow the session will be cleaned once the dialogue has ended. 
        '''
        # if cleaning, this is the last turn
        prompt_str = 'DIALOGUE_SERVER_MESSAGE:' + msg
        reply = prompt_str  # just send a text object when cleaning. 
        return reply, prompt_str 
    
    def recognition_failed_response(self, msg='', session_id=None, isbargein=None):
        '''Handles response to all errors that occur. Note msg is just a string - ie the errors are not formatted robustly as class
        objects or similar.  
        '''
        if isbargein is None:
            isbargein = self.allowbargein  # default 
        prompt_str = 'DIALOGUE_SERVER_ERROR:' + msg
        #lmr46, the following line is wrong! Errors should not be prompt to the user!!!
        #reply = self.question(question=prompt_str, session_id=session_id, isbargein=isbargein)
        reply=prompt_str
        return reply, prompt_str
    
    def prompt(self,prompt, session_id, isbargein=None, isfinal=False, isdtmf=False, dtmfsize=1):
        '''
        Create a prompt, for the moment the arguments are

        :param prompt: the text to be prompt
        :param isbargein: if barge in is allowed
        :param isfinal:  if it is the final sentence before the end of dialogue
        :return: reply in json
        '''
        if isbargein is None:
            isbargein = self.allowbargein  # default 
        reply={}
        reply['session']=session_id
        reply['replyType']="prompt"
        reply['bargein']=str(isbargein).lower()
        reply["final"]=str(isfinal).lower()
        reply["dtmf"]=str(isdtmf).lower()
        reply["dtmfsize"]=str(dtmfsize)
        reply['text'] = self._clean_text(prompt)
        return json.dumps(reply, ensure_ascii=False)

    def question(self,question, session_id, isbargein=None):
        """ TTS prompt and expect a speech reply from user.
        """
        if isbargein is None:
            isbargein = self.allowbargein  # default 
        reply={}
        reply['session']=session_id
        reply['replyType']="question"
        reply['bargein']=str(isbargein).lower()
        reply['text'] = self._clean_text(question)
        return json.dumps(reply,ensure_ascii=False)
    
    def control(self,return_control, session_id, kept_alive):
        '''
        Create a control message, for the moment the arguments are

        :param return_control: boolean indicating whether to return control or not
        :param session_id: the session id
        :param kept_alive: boolean indicating whether session is kept alive or not
        :return:
        '''
        reply={}
        reply['session']=session_id
        reply['replyType']="control"
        reply['returnControl']=str(return_control)
        reply['keptAlive']=str(kept_alive)
        return json.dumps(reply, ensure_ascii=False)

    def cleaningCNet(self,data):
        #lina
        '''
        Prunes the Confusion Network
        
        .. warning:: Not perfect:  improve it according to c++ pruning for now it is removing the paths where the !NULL is \
                    the most probable ...to be checked **See HRec.cpp Method: TranscriptionFromConf Line 4075**  (lmr46)

        :param data: json data
        :return:
        '''
        nbest={}
        pos=0
        #It should be a list of tuples
        tenBestSents=[]
        result= data["result"]
        if not "spans" in result:
            for alt in result["alts"]:
                strutt = alt["transcript"]
                if strutt!="!NULL":
                    tenBestSents.append((strutt,alt["confidence"]))
            return tenBestSents

        for sp in result["spans"]:
            _list=[]
            for alt in sp["alts"]:
                alt_word = alt["word"]
                alt_prob = alt["prob"]
                position=0
                for (i,j) in _list:
                    if alt_prob > j :
                        break
                    position+=1

                if position==len(_list):
                    _list.append((alt_word,alt_prob))
                else:
                    _list.insert(position,(alt_word,alt_prob))

                nbest[pos]=_list
            #print nbest[pos]
            pos+=1
        tmpBest={}

        j=0
        for i in range(len(nbest)):
            _list=nbest[i]
            if str(_list[0][0])=="!NULL":
                continue
            tmpBest[j]=nbest[i]
            j+=1

        nbest=tmpBest

        recPaths=[]
        recPaths=self.getNBest(nbest,len(nbest),[],[],recPaths)
        #print len(recPaths)
        tenBest= recPaths[0:10]

        for _list, like in tenBest:
            strutt=""
            for i in _list:
                word= str(i)
                if word != "!NULL" and word != '<s>' and word != '</s>':
                    strutt+=' '+ word
            logger.debug("CNET: strut: " +str(strutt))
            tenBestSents.append((strutt,like))

        logger.debug("CNET: top 10 sentences: "+str(tenBestSents))
        return tenBestSents

    def getNBest(self, nbest, key, words, like, retPaths):
        '''
           Returns the N-Best list from CNet candidates
        '''
        if key == 0:
            position=0
            loglike= sum(like)
            for (i,j) in retPaths:
                if j< loglike:
                    break
                position+=1
            if position ==  len(retPaths):
                retPaths.append((words[:], loglike))
            else:
                retPaths.insert(position,(words[:], loglike))
            return retPaths

        i=0
        for alt in nbest[key-1]:
            if len(words)<len(nbest):
                words.insert(0,alt[0])
                like.insert(0,alt[1])
            else:
                words[key-1]=alt[0]
                like[key-1]=alt[1]

            if i > 1:
                break

            self.getNBest(nbest,key-1,words,like,retPaths)
            i+=1

        return retPaths

    def _clean_text(self,RAW_TEXT):
        """
        """
        # The replace() is because of how words with ' come out of the Template SemO.
        JUNK_CHARS = ['(',')','{','}','<','>','"',"'"]
        return ''.join(c for c in RAW_TEXT.replace("' ","") if c not in JUNK_CHARS)



#================================================================================================
# MAIN FUNCTION 
#================================================================================================
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='DialogueServer')
    parser.add_argument('-C','-c', '--config', help='set config file', required=True, type=argparse.FileType('r'))
    parser.set_defaults(use_color=True)
    parser.add_argument('--nocolor', dest='use_color',action='store_false', help='no color in logging. best to\
                        turn off if dumping to file. Will be overriden by [logging] config setting of "usecolor=".')
    parser.add_argument('-s', '--seed', help='set random seed', type=int)
    args = parser.parse_args()

    seed = Settings.init(config_file=args.config.name, seed=args.seed)
    ContextLogger.createLoggingHandlers(config=Settings.config, use_color=args.use_color)
    logger.info("Random Seed is {}".format(seed))
    Ontology.init_global_ontology()

    # RUN THE DIALOGUE SERVER:
    dial_server = DialogueServer()
    #dial_server.postToCamdial(1234,"voip-555556789")       # just test msg posting to camdial 
    dial_server.run()



#END OF FILE
