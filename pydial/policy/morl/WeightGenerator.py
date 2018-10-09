'''
Created on 19 Oct 2016

@author: su259
'''
from utils import Settings, ContextLogger
logger = ContextLogger.getLogger('')

class WeightGenerator(object):
    '''
    classdocs
    '''


    def __init__(self, domainString, weightGeneration = None, distribution = None):
        self.domainString = domainString
        
        self.probs = []
        self.weightList = []
        total = 1

        if weightGeneration is None:
            weightGeneration = "auto" # auto or hdc
        if distribution is None:
            distribution = "uniform" # valley or uniform
        
        if weightGeneration == "auto":
            for i in range(101):
                self.weightList.append(float(i)/100.0)
        elif weightGeneration == "hdc":
            self.weightList = [0.1,0.3,0.5,0.7,0.9]
                

        if distribution == "uniform":
            self.probs = [1] * len(self.weightList)
            total = len(self.weightList)
        elif distribution == "valley":
            # works only for weightLists of uneven length
            value = len(self.weightList)-1
            middle = len(self.weightList)/2
            down = True
            total = 0

            for _ in range(len(self.weightList)):
                self.probs.append(value)
                total += value

                if down:
                    value -= 1
                    if value == middle:
                        down = False
                else:
                    value += 1

        self.probs = map(lambda a: float(a)/float(total),self.probs)
        
    def updateWeights(self, weights = None):
        if weights is None:
            weights = self._getWeightsPair()
        Settings.config.set("mogp_"+self.domainString,"weights","{0:.2f} {1:.2f}".format(weights[0],weights[1]))
        logger.info("MOGP weights set to {}".format(weights))
        
    
    def _getWeightsPair(self):
        w = [0,0]

        w[0] = Settings.random.choice(self.weightList, p=self.probs)
        w[1] = 1-w[0]
        
        return w 