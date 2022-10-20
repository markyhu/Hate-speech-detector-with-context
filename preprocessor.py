import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import *
import re

from collections import Counter


nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words("english")
stemmer = PorterStemmer()




def rm_spaces(text):
    """
    Accepts a text string and turn multiple white spaces into a single white space.
    """
    space_pattern = '\s+'
    
    text = re.sub(space_pattern, ' ', text)
    
   
    return text
    


def tokenize(text):
    """
    Removes non-alphanumeric and special characters, sets to lowercase, remove stopwords and then splits into tokens.
    """
    pattern = re.compile('([^\s\w]|_)+') ## Strip out any non-alphanumeric, non-whitespaces
    clean_text = pattern.sub('', text).lower()
    tokens = [t for t in clean_text.split() if t not in stopwords]
    return tokens



def cleanStr(string):
    """
    Remove extra spaces, then remove special characters 
    and expand some contractions.
    """
    space_pattern = '\s+'
    string = re.sub(space_pattern, ' ', string)
    string = re.sub(r"[^A-Za-z0-9(),!?'\'\`\"]", " ", string)
    string = re.sub(r"ain\'t", " are not ", string)
    string = re.sub(r"\'m", " am ", string)
    string = re.sub(r"\'s", " is ", string) 
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"``",' \' ',string)
    string = re.sub(r"`",' \' ',string)
    return string.strip()
 
 

#unigram,bigram,trigram counts
def frenquency(n,tokens):
    
    return Counter(nltk.ngrams(tokens,n))



