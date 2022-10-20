import lstm
import numpy as np
np.random.seed(1008)
import tensorflow as tf
tf.random.set_seed(1234)
import nn_utils
from preprocessor import cleanStr 
import gensim.downloader as api
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences




#embedding_model = api.load('word2vec-google-news-300')
embedding_model = api.load('glove-wiki-gigaword-300')
embedding_dim = len(embedding_model['dog'])

#model path
saveJ = './models/blstm.json'
saveW = './models/blstm-weights.h5' 


#load data as list
print('Loading data...')

data_set = nn_utils.loadJson('./corpus/fox-news-comments.json')


A = []
P = []
R = []
F = []




lstm_outputs =100  # dimension of lstm layer
dense_outputs = 256 # dimension of dense layer
batches = 64 # batch size
# max_sequence_len = max(len(cleanStr(sent['text']).split()) for sent in data_set)
sequence_len =150 #use most frequent sequence length
folds =5
weights ={0:2.5,1:1} #class weights

#Collect corpus from target and previous comments
corpus = [i['text'] for i in data_set]
context_corpus = [i['prev'] for i in data_set]
tokenizer = Tokenizer()
tokenizer.fit_on_texts([cleanStr(sent) for sent in context_corpus])

context_tokenizer = Tokenizer()
context_tokenizer.fit_on_texts([cleanStr(sent) for sent in context_corpus])


word_index = tokenizer.word_index
index_word = tokenizer.index_word
vocab = len(word_index)+1 #vocab size


word_index_context = context_tokenizer.word_index
index_word_context = context_tokenizer.index_word
vocab_context = len(word_index_context)+1 #context vocab size

#Create emebedding matrix from pretrained embeddings
embedding_matrix = nn_utils.create_embedding_matrix(index_word,embedding_model)
embedding_matrix_context = nn_utils.create_embedding_matrix(index_word_context,embedding_model)

#Crosss validation
for fold in range(folds):

    (train, test) = nn_utils.split_file(data_set, fold)

    #Prepare input for lstm
    train_data,y_train = [i['text'] for i in train], [i['label'] for i in train]
    test_data,y_test = [i['text'] for i in test], [i['label'] for i in test]
    train_data= tokenizer.texts_to_sequences(train_data)
    test_data = tokenizer.texts_to_sequences(test_data)
    
    X_train = pad_sequences(train_data,maxlen= sequence_len,truncating='post',padding='post')
    X_test = pad_sequences(test_data,maxlen= sequence_len,truncating='post',padding='post')
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    train_prev,test_prev = [i['prev'] for i in train],[i['prev'] for i in test]
    train_prev,test_prev= context_tokenizer.texts_to_sequences(train_prev),context_tokenizer.texts_to_sequences(test_prev)
    X_train_prev = pad_sequences(train_prev,maxlen=sequence_len,truncating='post',padding='post')
    X_test_prev = pad_sequences(test_prev,maxlen=sequence_len,truncating='post',padding='post')

    print('Build model...')
    #model=rnn.blstm(lstm_outputs,dense_outputs,vocab,embedding_dim,sequence_len,embedding_matrix)
    model=lstm.blstm_context(lstm_outputs,dense_outputs,vocab,vocab_context,sequence_len,embedding_matrix,embedding_matrix_context)
    
    model.summary()
    print('Begin CV...')
    model.fit([X_train,X_train_prev],y_train,class_weight = weights ,batch_size=batches,validation_data=([X_test,X_test_prev],y_test),epochs=20,verbose=2)
    #print(history.history.keys())
    y_predict = model.predict([X_test,X_test_prev],batch_size = batches)

    # pred = open('./predictions/lstm-%d.txt' % fold , 'w')
    # for i in range(len(y_predict)):
    #   pred.writelines(str(y_predict[i][0])+'\n')

    a,p,r,f = nn_utils.metrics(y_test,y_predict)
    print('Accuracy:\tPrecision:\tRecall:\tF-score:') 
    print('%f\t%f\t%f\t%f\t%f'%(a, p, r, f))	
    # rnn.save_model(model,saveJ,saveW)
    A += [a]
    P += [p]
    R += [r]
    F += [f]


#Mean values of metrics
print('Overall')
print('%f\t%f\t%f\t%f\t%f' %(sum(A)/len(A),sum(P)/len(P),sum(R)/len(R),sum(F)/len(F)))