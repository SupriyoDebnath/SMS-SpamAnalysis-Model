import sys
import re
import numpy
import pandas
import nltk
import spacy
import sklearn
import joblib

from Normalizer import TextNormalizer
from Vectorizer import TextVectorizer

# check imports
print('Python: %s' % (sys.version))
print('NumPy: %s' % (numpy.__version__))
print('Pandas: %s' % (pandas.__version__))
print('Joblib: %s' % (joblib.__version__))
print('NLTK: %s' % (nltk.__version__))
print('SpaCy: %s' % (spacy.__version__))
print('Scikit-learn: %s' % (sklearn.__version__))

# test inputs
smscontent01 = 'Nah I don\'t think he goes to usf, he lives around here though'
smscontent02 = 'I\'d like to tell you my deepest darkest fantasies. Call me 09094646631 just 60p/min. To stop texts call 08712460324 (nat rate)'

# serialize data from Input
features = pandas.Series([smscontent01, smscontent02])
print(features)

# normalize feature
normalizer = TextNormalizer(features)
features = normalizer.normalize()
print(features)

# vectorize feature
vectorizer = TextVectorizer(features, False)
features = features.apply(lambda x: vectorizer.transform(x))
print(features)

# load the nltk model from SMSSpamFilter_model.pkl
model_smsspamfilter = joblib.load('SMSSpamFilter_model.pkl')

# predict 0 - ham, 1 - spam
predictions = features.apply(lambda x: 'ham' if model_smsspamfilter.classify(x) == 0 else 'spam')
print(predictions)



    

