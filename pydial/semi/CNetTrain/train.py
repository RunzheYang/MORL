import ConfigParser, sys
import Classifier, sutils

def train(config):
    c = Classifier.classifier(config)
    dataroot = config.get("train", "dataroot")
    dataset = config.get("train", "dataset")
    dw = sutils.dataset_walker(dataset = dataset, dataroot=dataroot, labels=True)
    c.train(dw)
    c.save(config.get("train", "output"))
    
def usage():
    print "usage:"
    print "\t python train.py config/eg.cfg"


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
    
    train(config)