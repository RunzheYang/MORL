import sys
sys.path.append("src/lib")
sys.path.append("lib")
from subprocess import call

import  os, ConfigParser, json


def evaluate(config) :
    dataroot = config.get("decode", "dataroot")
    dataset = config.get("decode", "dataset")
    scripts_folder = os.path.join(dataroot, os.pardir, "scripts")
    score_script = os.path.join(scripts_folder, "score_slu.py")
    
    call(["python", score_script, "--dataset="+dataset,
        "--dataroot="+dataroot,  "--decodefile="+config.get("decode","output"),
        "--scorefile="+config.get("evaluate","csv_output"),
        "--ontology="+config.get("grammar","ontology"),
        "--trackerfile="+config.get("evaluate","tracker_output"),
        ])
    print "created ", config.get("evaluate","csv_output")
    
    score_script = os.path.join(scripts_folder, "score.py")
    call(["python", score_script,
          "--dataset="+dataset,
          "--dataroot="+dataroot,
          "--trackfile="+config.get("evaluate","tracker_output"),
          "--scorefile="+config.get("evaluate","tracker_csv_output"),
          "--ontology="+config.get("grammar","ontology"),  
    ])
    print "created", config.get("evaluate","tracker_csv_output")
    
if __name__ == '__main__':
    if len(sys.argv) != 2 :
        print "Usage : "
        print "python evaluate.py discriminative/config/example.cfg"
        sys.exit()
        
    config = ConfigParser.ConfigParser()
    try :
        config.read(sys.argv[1])
    except Exception as inst:
        print "Failed to parse file", inst
    evaluate(config)
