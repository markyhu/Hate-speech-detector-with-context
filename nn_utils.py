import numpy as np
np.random.seed(1008)
import random
random.seed(1280)
import tensorflow as tf
tf.random.set_seed(1234)

import random 
import json  
from sklearn import metrics as SKM
from sklearn.model_selection import KFold
from preprocessor import cleanStr 



def loadJson(file):
    """
    Load json file and return a list
    """
    data = []
    
    with open(file,'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    return data


def split_file(f_train,fold):
    """
    Split data 5 times, return train and test data for a specific fold
    """

    x_train = [] 
    y_train = []

    x_test = []
    y_test = []

    cv = 5

    kf = KFold(n_splits= cv , shuffle=True, random_state=123)
    T = []
    for train_index, test_index in kf.split(f_train):
        T += [(train_index, test_index)]

    train_index = T[fold][0]
    test_index  = T[fold][1]

    train_ = []
    test_ = []

    for i in train_index:
        train_ += [f_train[i]]


    for i in test_index:
        test_ += [f_train[i]]
 

    return train_, test_
 


 
def create_embedding_matrix(id_to_word,embed_model):
    """
    Accepts a mapping of word ids to words from a vocabulary and pretrained embedding,
    returns an embedding matrix for this vocabulary.
    """
    vocab_size = len(id_to_word)+1
    embed_size = len(embed_model['dog'])      
    embedding_matrix = np.zeros((vocab_size,embed_size))
    oov = 0
    drange = np.sqrt(6. / (vocab_size + embed_size))  

    for i in id_to_word:
        if(id_to_word[i] not in embed_model):
            #Initialize embeddings for out of vocabulary words 
            embedding_matrix[i] = np.random.uniform(low=-drange, high=drange, size=(embed_size,))  
            oov +=1 
        else:
            embedding_matrix[i] = embed_model[ id_to_word[i] ] 

    print ('Created embedding matrix, OOVs = ',oov)
    return embedding_matrix





def metrics(y_test,y_predict):
    """
    This function takes the true labels and predicted labels and returns the accuaracy, precision, recall and f1 score.
    precision,recall and f1-score are calculated for positive class only.
    """ 
    if(len(y_test)!=len(y_predict)):
        return None
    y_predict_b = [ ]
    y_predict_r = [ ]
    for i in range(len(y_test)):
        y_predict_b += [1 if(y_predict[i][0]>=0.5) else 0]
        y_predict_r += [y_predict[i][0]]

    acc = SKM.accuracy_score(y_test,y_predict_b)
    pre = SKM.precision_score(y_test,y_predict_b,pos_label=1)
    recall = SKM.recall_score(y_test,y_predict_b,pos_label=1)
    f = SKM.f1_score(y_test,y_predict_b,pos_label=1)

    return acc,pre,recall,f