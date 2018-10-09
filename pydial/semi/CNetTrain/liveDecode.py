# outputs a decode file based on the live system output
import sys, sutils, json

def usage() :
    print "Usage:"
    print "python liveDecode.py dataroot dataset output_file"
    
if len(sys.argv) != 4 :
    usage()
    sys.exit()
    
dataroot, dataset, output_fname = sys.argv[-3:]
dw = sutils.dataset_walker(dataset = dataset, dataroot=dataroot)

output = {
    "dataset": dw.datasets,
    "sessions": []
    }

for call_num, call in enumerate(dw):
    session = {"session-id" : call.log["session-id"], "turns":[]}
    for log_turn, _ in call:
        live_hyps = log_turn["input"]["live"]["slu-hyps"]
        if len(live_hyps) == 0:
            live_hyps = [
                {
                    'slu-hyp':[{'slots':[], 'act':'null'}],
                    'score':1.0        
                }
            ]
        session["turns"].append({
            "slu-hyps": live_hyps
        })
    output["sessions"].append(session)
            
output_file = open(output_fname, "wb")
json.dump(output, output_file, indent=4)
output_file.close()