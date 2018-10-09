import ConfigParser, sys
import Classifier, sutils

def export(config):
    c = Classifier.classifier(config)
    c.load(config.get("train", "output"))
    c.export(config.get("export","models"),
             config.get("export","dictionary"),
             config.get("export","config"))
    
def usage():
    print "usage:"
    print "\t python export.py config/eg.cfg"


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
    
    export(config)