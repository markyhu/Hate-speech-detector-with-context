
import re
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from preprocessor import rm_spaces,cleanStr
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate,KFold




sentiment_analyzer = VS()
nltk.download('averaged_perceptron_tagger') # for POS tagging

def pos_tagger(text):
    """
    Accepts a text string and returns a list of the part of speech tags for each word in the text.
    """
    tokenized_text  = cleanStr(text).split()
    tag_list = [t[1] for t in nltk.pos_tag(tokenized_text)]
    tags = ' '.join(tag_list)
    return tags




#Get other features

def other_features(text):
    """
    This function takes a string and returns a list of features.
    These include sentiment scores, some surface features from the text.
    """
   
    sentiment = sentiment_analyzer.polarity_scores(text) #Sentiment scores
    
    sentence = rm_spaces(text) #Remove extra spaces
    
    
    clean_sentence = cleanStr(text) #strip out non-alphanumeric
   
    num_chars = sum(len(w) for w in sentence)
    
    
    num_words = len(clean_sentence.split())
   
    num_unique_terms = len(set(sentence.split()))
    
    num_special_chars = len(re.sub('([\s\w]|_)+','',sentence))
   


    features = [ num_chars, num_words,num_special_chars,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                ]
    return features





def cross_val(clf,X,y):
    """
    This function takes a classifier, X and y and returns the
    mean values of metrics and the coeffecients of the classifier from each fold.
    """

    scoring = {'accuracy' : make_scorer(accuracy_score), 
            'precision' : make_scorer(precision_score),
            'recall' : make_scorer(recall_score), 
            'f1_score': make_scorer(f1_score)}
    coef = []

    print('Begin CV...')
    cv = KFold(n_splits= 5 , shuffle=True, random_state=123)
    cv_scores = cross_validate(clf,X,y,scoring=scoring, cv = cv,return_estimator = True)

    for model in cv_scores['estimator']:
      coef.append(model.coef_)
    
    
    
    return {k:v.mean() for k,v in cv_scores.items() if k != 'estimator'},coef
