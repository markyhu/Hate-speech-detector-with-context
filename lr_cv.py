
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from preprocessor import tokenize,cleanStr
from lr_utils import pos_tagger,other_features,cross_val

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Load data
dataset = pd.read_json('./corpus/fox-news-comments.json',lines=True)
n_sample = dataset.shape[0]
target_comments = dataset['text']
prev_comments =dataset['prev']
label = dataset['label']


#Vectorizer for word-level unigrams, bigrms, trigrams
wd_tfidf_vectorizer = TfidfVectorizer(
    analyzer = 'word',
    #tokenizer=tokenize,
    preprocessor=cleanStr,
    lowercase=False,
    ngram_range=(1, 3),
    #stop_words=stopwords,
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    min_df = 3,
    max_df= 0.8

)

#Vectorizer for characters
chr_tfidf_vectorizer = TfidfVectorizer(
    analyzer = 'char',
    preprocessor=cleanStr,
    lowercase=False,
    ngram_range=(3, 6),
    #stop_words=stopwords,
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    min_df  =3,
    max_df =0.8
)

#Vectorizer for POS tags
pos_tfidf_vectorizer = TfidfVectorizer(
    tokenizer=None,
    lowercase=False,
    preprocessor=None,
    ngram_range=(2, 4),
    stop_words=None,
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace'
   )

#POS count distribution
# pos_count_vectorizer = CountVectorizer(lowercase=False,tokenizer=None,preprocessor=None)
# pos_count = pos_count_vectorizer.fit_transform(target_comments.apply(pos_tagger)).toarray()


#Get tfidf matrix for target comments 
wd_tfidf = wd_tfidf_vectorizer.fit_transform(target_comments).toarray()
chr_tfidf = chr_tfidf_vectorizer.fit_transform(target_comments).toarray()
pos_tfidf = pos_tfidf_vectorizer.fit_transform(target_comments.apply(pos_tagger)).toarray()

#Get tfidf matrix for previous comments
pre_wd_tfidf = wd_tfidf_vectorizer.fit_transform(prev_comments).toarray()
pre_chr_tfidf = chr_tfidf_vectorizer.fit_transform(prev_comments).toarray()
pre_pos_tfidf = pos_tfidf_vectorizer.fit_transform(prev_comments).toarray() 



#Prepare inputs
other_feat= np.array([other_features(t) for t in target_comments])
context_other_feat = np.array([other_features(t) for t in prev_comments])
wd = wd_tfidf
char = chr_tfidf
wd_char = np.concatenate((wd_tfidf,chr_tfidf),axis=1) 
wd_char_whole = np.concatenate((wd_tfidf,chr_tfidf,pos_tfidf,other_feat),axis=1) 
wd_context = np.concatenate((wd_tfidf,pre_wd_tfidf),axis=1)
char_context = np.concatenate((chr_tfidf,pre_chr_tfidf),axis=1)
wd_char_context = np.concatenate((wd_tfidf,chr_tfidf,pre_wd_tfidf,pre_chr_tfidf),axis=1)
wd_char_whole_context= np.concatenate((wd_tfidf,chr_tfidf,pos_tfidf,other_feat,pre_wd_tfidf,pre_chr_tfidf,pre_pos_tfidf,context_other_feat),axis=1) #with previous comment as context features


#Cross validation for 8 logistic regression models
input_list= [wd,wd_context,char,char_context,wd_char,wd_char_context,wd_char_whole,wd_char_whole_context]
clf = LogisticRegression(C=0.01, max_iter = 100000,penalty='l2',class_weight='balanced')
cv_results = []

for input in input_list:
    cv_scores,coef = cross_val(clf,input,label)
    cv_results.append(cv_scores)

print(cv_results)










