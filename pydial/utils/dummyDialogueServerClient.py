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
dummyDialogueServerClient.py - script for talking to the dialogue server via cmd line, i.e. without using any voice client.
===========================================================================================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

**Basic Usage**:
    Also can be used to simulate multiple concurrent calls to a running DialogueServer() - each of these is run in it's own
    process via python's multiprocessing module.
    >> python dummyDialogueServerClient.py fake number_of_fake_clients

    --- where number_of_fake_clients is an integer.

    NB: This was just a tool I wrote for developing DialogueServer, but it is actually pretty useful so i've put it in the repo and it
    is also used by nosetests now to execute some basic testing of DialogueServer.

************************

'''


__author__ = "cued_dialogue_systems_group"

import httplib
import json
import socket
import numpy as np
import multiprocessing as mp
import pprint
pp = pprint.PrettyPrinter(indent=2)


class fake_client():
    '''
    Text based client for interacting with DialogueServer via the cmd line
    '''
    def __init__(self, user_in, hostname='localhost', port=8082, numDialogs=2):
        
        self.USER = user_in
        self.dtmf_num = {False: '30003', True: '1'}   # for task and then subjective feedback
        self.BYE_AFTER = 3
        self.SLEEP_HIGH = 10  # 1/10 seconds
        self.DO_N_DIALOGS = numDialogs
        self.DONE_DIALOGS = -1
        self.random = np.random.RandomState()
        self.hostname = hostname
        self.port = port
        self.reset()
        
    def reset(self):
        '''
        Resets dialogue state.
        '''
        self.dtmf_count = 0 
        self.turns = 0 
        self.ID = str(self.random.randint(low=1000,high=9999))
        self.DONE_DIALOGS += 1
        
    def talk(self, output=None, delay=None):
        #conn = httplib.HTTPConnection('178.79.137.90',8082)
        if delay is not None:
            import time
            time.sleep(delay)
        conn = httplib.HTTPConnection(self.hostname, self.port)
        try:
            while 1:
                
                params = self.gen_json_response()
                headers = {'content-type': 'application/json'}
                #response = requests.post('http://localhost:8082', data=json.dumps(params), headers=headers)
                
                #request command to server
                if self.USER:
                    user_int = int(raw_input('1=newcall or 2=next: '))
                    REQUEST = 'newcall' if user_int == 1 else 'next'
                else:
                    REQUEST = 'newcall' if self.turns == 1 else 'next'
                    time.sleep(np.random.randint(low=1,high=self.SLEEP_HIGH)/10.0)
                
                # Attempt to communicate with DialogueServer
                #print ">>>>>>> "+str(self.turns)+" POST", '/'+REQUEST
                #pp.pprint(params)
                try:
                    conn.request('POST', '/'+REQUEST, json.dumps(params),headers)
                except socket.error:
                    exit('socket.error - Connection refused. Have you launched a DialogueServer? Are the hostname and port ok?')
                
                #get response from server
                rsp = conn.getresponse()
                #print "<<<<<<<< REPLY"
                #pp.pprint(rsp.status)
                #pp.pprint(rsp.reason)
                #pp.pprint(rsp.read())
                if self.turns >= self.BYE_AFTER and self.dtmf_count == 2:
                    # reset as we are done
                    self.reset()
                    if self.DONE_DIALOGS == self.DO_N_DIALOGS:
                        break
                
                #print server response and data
                if output is not None:
                    stream = str(rsp.status) + ' ' + str(rsp.reason) + ' ' + str(rsp.read())
                    output.put(stream)
                else:
                    print(rsp.status, rsp.reason)
                    data_received = rsp.read()
                    print(data_received)
        except KeyboardInterrupt:
            print '\nCLOSING CLIENT'

    def gen_json_response(self):
        self.turns += 1
        
        if self.USER:
            # get user input from keyboard
            _inputtype = int(raw_input("ENTER 1:ASR, 2:DTMF: "))
            if _inputtype == 1:                 # _inputtype == 1 indicates source is asr
                asr_input = raw_input('ASR: ')
            elif _inputtype == 2:               # _inputtype == 2 indicates source is dtmf
                dtmf_input = raw_input('DTMF (ie 30003): ')
        else:
            # synthesise user input
            if self.turns > self.BYE_AFTER:
                _inputtype = 2
                dtmf_input = self.dtmf_num[self.dtmf_count]
                self.dtmf_count += 1    # becomes True 
            else:
                _inputtype = 1
                asr_input = 'i want a cheap hotel' if self.turns < self.BYE_AFTER else 'bye'

        
        if _inputtype == 1:
            asr_json = { "session": "local-audio-160303_"+self.ID, 
                       "result" : { "resultType":"Partial" ,
                        "alts": [
                          {"transcript" : asr_input, "confidence" : 10.0 },
                          #{"transcript" : "Ralph Nader hotel ", "confidence" : -1 } ,
                          #{"transcript" : "Ralph Nader hertz how ", "confidence" : -1 } ,
                          #{"transcript" : "uhhuh Ralph Nader hurts how ", "confidence" : -1 } ,
                          {"transcript" : "rhubarb rhubarb", "confidence" : -1 }
                        ]
                          #----------
                          # NOTE: if you add in the extra uncertainty of more asr hyps by uncommenting above lines
                          # you can see the focus and baseline trackers are quite slow to add on probability mass to belief state.
                          }
                        }
            if self.turns == 1:
                # below is the information DIALPORT sends us - basically just what domain to start in. 
                asr_json["context"] =  { "init_domain": [{ "value": "CamHotels", "confidence": 0.6}, 
                                             { "value": "SFHotels", "confidence": 0.1}],
                            "entities": [{"type": "LOC", "value": "pittsburgh"}]                  
                            }
            return asr_json
        elif _inputtype == 2:
            # TODO  -- figure what DTMF json looks like
            dtmf_json =  { "session": "local-audio-160303_"+self.ID, 
                       "result" : { "dtmf" : dtmf_input} 
                       }
                        
            return dtmf_json

def run_fake_clients(NUM_CLIENTS=1, DIALOGS_PER_CLIENT=1, pause_time=None):
    # PUT THE FAKE CLIENTS IN multiprocessing:
    output = mp.Queue()
    processes = []
    clients = {}
    for c in xrange(NUM_CLIENTS):
        clients[c] = fake_client(user_in=False,numDialogs=DIALOGS_PER_CLIENT)
        processes.append( mp.Process(target=clients[c].talk, args=(output,pause_time)) )  
    for p in processes: p.start()
    for p in processes: p.join()
    # Get process results from the output queue
    outp = []
    while not output.empty():
        outp.append(output.get())
    return outp

def main():
    import sys
    USAGE = "USAGE: python dummyDialogueServerClient.py 'user'/'fake' [num_clients] [hostname] [port]"
    if len(sys.argv) < 2:    
        exit(USAGE)
    
    #TODO - add cmd line input of hostname and port
    
    if sys.argv[1] == 'user':
        client = fake_client(user_in=True)
        client.talk()
    elif sys.argv[1] == 'fake':
        try:
            NUM_CLIENTS = int(sys.argv[2])      
        except:
            print "Make sure to give the number of fake clients as well if using 'fake' option"
            exit(USAGE)
        run_fake_clients(NUM_CLIENTS)
        
    else:
        print "Argument one must be either 'user' or 'fake'"
        exit(USAGE)


if __name__ == '__main__':
    main()

#END OF FILE
