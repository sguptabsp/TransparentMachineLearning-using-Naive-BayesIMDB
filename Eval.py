import numpy as np
class Eval:
    def __init__(self, pred, gold):
        self.pred = pred
        self.gold = gold
        
    def Accuracy(self):
        return np.sum(np.equal(self.pred, self.gold)) / float(len(self.gold))

    def EvalPrecition(self):
        pos = np.argwhere(self.gold == 1).flatten()
        neg = np.argwhere(self.gold == -1).flatten()
        TP = np.sum(np.equal(np.array(self.pred)[pos], np.array(self.gold)[pos]))
        FP = np.sum(np.not_equal(np.array(self.pred)[neg], np.array(self.gold)[neg]))
        return TP/(TP+FP)

    def EvalRecall(self):
        pos = np.argwhere(self.gold == 1).flatten()
        TP = np.sum(np.equal(np.array(self.pred)[pos], np.array(self.gold)[pos]))
        FN = np.sum(np.not_equal(np.array(self.pred)[pos], np.array(self.gold)[pos]))
        return TP/(TP+FN)
