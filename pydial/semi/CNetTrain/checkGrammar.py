# this will check the [grammar] section in your config file against the examples
# found in the train and test set

import ConfigParser, sys, json
import Tuples, sutils
from collections import defaultdict

def checkGrammar(config):
    t = Tuples.tuples(config)
    
    all_acts = set([])
    all_slots = set([])
    
    complaints = defaultdict(lambda :set([]))
    for section in ["train", "test"]:
        if not config.has_section(section) :
            continue
        print "Checking ", section
        dataroot = config.get(section, "dataroot")
        dataset = config.get(section, "dataset")
        dw = sutils.dataset_walker(dataset = dataset, dataroot=dataroot, labels=True)
        for call in dw:
            for log_turn, label_turn in call:
                uacts = label_turn["semantics"]["json"]
                for uact in uacts:
                    all_acts.add(uact['act'])
                    # check the act is in acts
                    if uact['act'] not in t.acts:
                        complaints["acts"].add(uact['act'])
                    # check if its empty, but in nonempty_acts
                    if uact['slots'] == [] and uact['act'] in t.nonempty_acts :
                        complaints["nonempty_acts"].add(uact['act'])
                    # check if its full, but in nonfull_acts
                    if uact['slots'] != [] and uact['act'] in t.nonfull_acts :
                        complaints["nonfull_acts"].add(uact['act'])
                    # check all the mentioned slots are in slots
                    for slot, value in uact['slots']:
                        if slot == "this" :
                            continue
                        all_slots.add(slot)
                        if slot == "slot" and uact['act'] == "request" :
                            all_slots.add(value)
                            if value not in t.slots :
                                complaints["slots"].add(slot)
                            continue
                        if slot not in t.slots_informable :
                            complaints["slots_informable"].add(slot)
                        if slot in t.slots_enumerated and value not in (t.ontology[slot]+['dontcare']) :
                            complaints["slots_enumerated"].add((slot,value))
    texts = {
        "acts":"Acts found that were not in acts",
        "nonempty_acts":"Empty acts found declared as nonempty",
        "nonfull_acts":"Full acts found declared as nonfull",
        "slots":"undeclared slots found ",
        "slots_informable":"undeclared informable slots found ",
        "slots_enumerated":"a slot,value pair not in the enumeration of slot"
             }
    for key in texts:
        if complaints[key] :
            print texts[key]
            print list(complaints[key])
    
    # check there weren't any unused acts:
    for act in t.acts:
        if act not in all_acts:
            print "act =",act,"unused"
    for slot in t.slots:
        if slot not in all_slots:
            print "slot =",slot,"unused"
    
            
        

def usage():
    print "usage:"
    print "\t python checkGrammar.py config/eg.cfg"


if __name__ == '__main__':
    if len(sys.argv) != 2 :
        usage()
        sys.exit()
        
    config = ConfigParser.ConfigParser()
    try :
        config.read(sys.argv[1])
    except Exception as e:
        print "Failed to parse file"
        print e
    
    checkGrammar(config)