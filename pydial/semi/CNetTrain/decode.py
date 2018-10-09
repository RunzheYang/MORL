import ConfigParser, sys
import Classifier, sutils


def decode(config):
    c = Classifier.classifier(config)
    c.load(config.get("train", "output"))
    
    dataroot = config.get("decode", "dataroot")
    dataset = config.get("decode", "dataset")
    dw = sutils.dataset_walker(dataset = dataset, dataroot=dataroot, labels=False)
    
    c.decodeToFile(dw, config.get("decode","output"))
    
def usage():
    print "usage:"
    print "\t python decode.py config/eg.cfg"

def init_classifier(config):
    c = Classifier.classifier(config)
    c.load(config.get("train", "output"))
    return c

def decode(c,config,sentinfo):
    slu_hyps=c.decode_sent(sentinfo, config.get("decode","output"))
    return slu_hyps

def testing_currTurn():
            sentinfo={
            "turn-id": 2,
            "asr-hyps": [
                    {
                        "asr-hyp": "expensive restaurant in south part of town",
                        "score": -0.512136
                    }
                    ,
                    {
                         "asr-hyp": "a restaurant in south part of town",
                         "score": -2.429358
                    },
                    {
                         "asr-hyp": "expensive expensive restaurant in south part of town",
                         "score": -2.429358
                    },
                    {
                         "asr-hyp": "expensive restaurant restaurant in south part of town",
                         "score": -3.210609
                    },
                    {
                         "asr-hyp": "expensive in south part of town",
                         "score": -3.211676
                    },
                    {
                         "asr-hyp": "expensive restaurant south part of town",
                         "score": -3.347932
                    },
                    {
                         "asr-hyp": "the expensive restaurant in south part of town",
                         "score": -4.268281
                    },
                    {
                         "asr-hyp": "a expensive restaurant in south part of town",
                         "score": -4.34658
                    },
                    {
                         "asr-hyp": "a restaurant restaurant in south part of town",
                         "score": -5.127831
                    },
                    {
                         "asr-hyp": "expensive expensive restaurant restaurant in south part of town",
                         "score": -5.127831
                    }
                ]
            }
            return sentinfo

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

    # decode(config)

    sentinfo=testing_currTurn()
    slu_hyps=decode(config, sentinfo)
    for hyp in slu_hyps:
        print hyp