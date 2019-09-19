import sys
from scipy.sparse import csr_matrix
import numpy as np
from Eval import Eval
from math import log, exp
import time
from imdb import IMDBdata
from Vocab import Vocab
import warnings
warnings.simplefilter('ignore')
from numpy import float128
import matplotlib.pyplot as plt

class NaiveBayes:
    def __init__(self, data, ALPHA=1.0):
        self.ALPHA = ALPHA
        self.data = data # training data
        #TODO: Initalize parameters
        self.vocab_len = data.vocab.GetVocabSize();
        self.count_positive = []
        self.count_negative = []
        self.num_positive_reviews = 0
        self.num_negative_reviews = 0
        self.total_positive_words = 0
        self.total_negative_words = 0
        self.P_positive = 0.5
        self.P_negative = 0.5
        self.deno_pos = 1
        self.deno_neg = 1
        self.like_pos = []
        self.like_neg=[]
        self.sum_like_pos=0
        self.sum_like_neg=0
        self.Train(data.X,data.Y)

    '''
    Train function find value for all Navie Bayes parameter
    
    Parameter:
    X : Feature matrix
    
    Y : Class labels array
    
    retrun:
    '''
    def Train(self, X, Y):
        

        #Unique words in training set
        #self.vocab_len = data.vocab.GetVocabSize();;

        #Find index value of Positive class
        positive_indices = np.argwhere(Y == 1.0).flatten()
        #Finf index vaue of Negative class
        negative_indices = np.argwhere(Y == -1.0).flatten()

        #List of words in vocab with their count in Positive class i.e. how many that word appeared in Positive class
        temp = X[positive_indices].sum(axis=0)
        self.count_positive = np.array(temp).flatten()
        #List of words in vocab with their count in Negative class i.e. how many that word appeared in Negative class
        temp = X[negative_indices].sum(axis=0)[0]
        self.count_negative = np.array(temp).flatten()

        # Total number of Positive review
        self.num_positive_reviews = len(positive_indices)
        # Total number of Negative review
        self.num_negative_reviews = len(negative_indices)

        # Total positive words in all Positive class
        self.total_positive_words = self.count_positive.sum()
        # Total negative words in all Negative class
        self.total_negative_words = self.count_negative.sum()

        # Positive Class Prior Probability
        self.P_positive = self.num_positive_reviews / len(Y);
        # Negative Class Prior Probability
        self.P_negative = self.num_negative_reviews / len(Y);
        
        #Denominator of Likelihood of Positive class
        self.deno_pos = self.total_positive_words+(self.ALPHA*self.vocab_len)
        #Denominator of Likelihood of Negative class
        self.deno_neg = self.total_negative_words+(self.ALPHA*self.vocab_len)
        
        #List of probability of Positive class words with laplace smoothing
        self.like_pos = np.array([(i+self.ALPHA)/(self.deno_pos) for i in self.count_positive])
        #List of probability of Negative class words with laplace smoothing
        self.like_neg = np.array([(i+self.ALPHA)/(self.deno_neg) for i in self.count_negative])
        
        #Denominator of Positive class Posterior probability i.e. Evidence
        self.sum_like_pos = np.sum(self.like_pos)*self.P_positive
        #Denominator of Negative class Posterior probability i.e Evidence
        self.sum_like_neg = np.sum(self.like_neg)*self.P_negative
        return

    '''
    PredictLabel function find label for given matrix.
    
    Parameter:
    X: Feature Matrix
    
    Return: Class label array
    '''
    def PredictLabel(self, X):
        
        pred_labels = []

        sh = X.shape[0]
        for i in range(sh):
            z = X[i].nonzero()
            #Initialize varible with Positive class prior probability
            posterior_prob_pos = float128(self.P_positive)
            #Initialize varible with Negative class prior probability
            posterior_prob_neg = float128(self.P_negative)
            #Calculating likelihood of Positive and Negative class 
            for j in z[1]:
                try:
                    posterior_prob_pos = posterior_prob_pos * np.power(self.like_pos[j],X[i][0,j])               
                    posterior_prob_neg = posterior_prob_neg * np.power(self.like_neg[j],X[i][0,j])           
                except ValueError:
                    pass
            #Normalize Probability
            prob_pos = posterior_prob_pos/(posterior_prob_pos+posterior_prob_neg)
            prob_neg = posterior_prob_neg/(posterior_prob_pos+posterior_prob_neg)
            
            if prob_pos>prob_neg: 
                #Predict Positive Class
                pred_labels.append(1.0)
            else:  
                #Predict Negative class
                pred_labels.append(-1.0)
        
        return pred_labels
    
    '''
    PredictLabelTh function find label for given matrix using threshold.
    Probability above threshold conside as Positive label.
    
    Parameter:
    X: Feature matrix
    
    threshold: Threshold value for probability
    
    Return: Class label array
    '''
    def PredictLabelTh(self, test, threshold):
        pred_labels = []

        sh = test.X.shape[0]
        for i in range(sh):
            #Calculating log probability of review
            prob = self.PredictProb(test,i)          
            if prob[2]>threshold: 
                #Predict Positive class
                pred_labels.append(1.0)
            else:
                #Predict Negative class
                pred_labels.append(-1.0)
        
        return pred_labels
    
    '''
    LogSum function return log sum exponential
    '''
    def LogSum(self, logx, logy):   
        m = max(logx, logy)        
        return m + log(exp(logx - m) + exp(logy - m))
    
    '''
    PredictProb function predict probability for given data and return class label
    
    Parameter:
    test : Feature matrix
    
    index: List of index for given matrix
    
    Return: 
    '''
    def PredictProb(self, test, indexes):
        
        i = indexes 
        predicted_label = 0
        z = test.X[i].nonzero()          
        posterior_prob_pos = 0
        posterior_prob_neg = 0
        #Calculating log of likelihood of positive and negative class
        for j in z[1]:
            try:
                posterior_prob_pos = posterior_prob_pos + log(np.power(self.like_pos[j],test.X[i][0,j]))               
                posterior_prob_neg = posterior_prob_neg + log(np.power(self.like_neg[j],test.X[i][0,j]))           
            except ValueError:
                pass            
        #Calculating log sum exp of evidence    
        evidence = self.LogSum(log(self.sum_like_pos),log(self.sum_like_neg))
        #Calculating log posterior probability of Positive class    
        predicted_prob_positive = log(self.P_positive) + (posterior_prob_pos) - evidence
        #Calculating log posterior probability of Negative class  
        predicted_prob_negative = log(self.P_negative) + (posterior_prob_neg) - evidence
        #Normalized Probability
        prob_positive = np.exp(float128(predicted_prob_positive))/(np.exp(float128(predicted_prob_positive))+np.exp(float128(predicted_prob_negative)))
        prob_negative = np.exp(float128(predicted_prob_negative))/(np.exp(float128(predicted_prob_positive))+np.exp(float128(predicted_prob_negative)))
        
        if prob_positive > prob_negative:
            #Predict Positive class 
            predicted_label = 1.0
        else:
            #Predict Negative Class
            predicted_label = -1.0

        return [test.Y[i], predicted_label, prob_positive, prob_negative]

    '''
    Evaluate performance on test data 
    '''
    def Eval(self, test):
        Y_pred = self.PredictLabel(test.X)
        # Y_pred = self.PredictLabel(test,len(test))
        ev = Eval(Y_pred, test.Y)
        # ev = Eval(Y_pred.pred, test.Y)
        return [ev.Accuracy(), ev.EvalPrecition(), ev.EvalRecall()]



if __name__ == "__main__":
    
    print("Reading Training Data")
    traindata = IMDBdata("%s/train" % sys.argv[1])
    print("Reading Test Data")
    testdata  = IMDBdata("%s/test" % sys.argv[1], vocab=traindata.vocab)
    
    print("Computing Parameters")
    Accuracy=[]
    for i in [0.1,0.5,1.0,5.0,10.0]:
        nb = NaiveBayes(traindata, float(i))
        evl = nb.Eval(testdata)
        Accuracy.append(evl[0])
        print('Alpha:', i , ' Accuracy:', evl[0], ' Precition:', evl[1], ' Recall:', evl[2])
    
    '''
    # plotting the points 
    plt.plot([0.1,0.5,1.0,5.0,10.0], Accuracy)
    # naming the x axis
    plt.xlabel('Alpha')
    # naming the y axis
    plt.ylabel('Accuracy')
    # giving a title to my graph
    plt.title('Alpha vs Accuracy')
    # function to show the plot
    plt.show()
    '''
    print("Log Probability of first 10 review")
    nb = NaiveBayes(traindata,float(sys.argv[2]))
    for i in range(10):
        res = nb.PredictProb(testdata,i)
        print(i, res)
    
    print("Predict Label with Threshold")
    precition=[]
    recall=[]
    
    for i in np.arange(0.1,1.0,0.1):
        pred = nb.PredictLabelTh(testdata,i)
        ev = Eval(pred, testdata.Y)
        precition.append(ev.EvalPrecition())
        recall.append(ev.EvalRecall())
        # ev = Eval(Y_pred.pred, test.Y)
        print('Threshold:', i, ' Accuracy: ', ev.Accuracy(), ' Precition: ',ev.EvalPrecition(), ' Recall: ',ev.EvalRecall())
    
    '''
    # Create some mock data
    t = recall
    data1 = precition
    data2 = np.arange(0.05, 1.0, 0.05)
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precition', color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Threshold', color=color)
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.show()
    '''
    
    print("Top 20 Positive words")
    list1=np.log(nb.like_pos)-np.log(nb.like_neg)
    list2=list(traindata.vocab.id2word.values())
    list1, list2 = (list(x) for x in zip(*sorted(zip(list1, list2), key=lambda pair: pair[0], reverse=True)))
    for i in range(20):
        print (list2[i] , list1[i])
    
    print("Top 20 Negative words")
    list1=np.log(nb.like_neg)-np.log(nb.like_pos)
    list2=list(traindata.vocab.id2word.values())
    list1, list2 = (list(x) for x in zip(*sorted(zip(list1, list2), key=lambda pair: pair[0], reverse=True)))
    for i in range(20):
        print (list2[i] , list1[i])