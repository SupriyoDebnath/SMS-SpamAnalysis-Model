import sys
import re
import numpy
import pandas
import nltk
import spacy
import sklearn
import joblib

from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import VotingClassifier

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.classify.scikitlearn import SklearnClassifier


# check imports
print('Python: %s' % (sys.version))
print('NumPy: %s' % (numpy.__version__))
print('Pandas: %s' % (pandas.__version__))
print('Joblib: %s' % (joblib.__version__))
print('NLTK: %s' % (nltk.__version__))
print('SpaCy: %s' % (spacy.__version__))
print('Scikit-learn: %s' % (sklearn.__version__))


# load dataset
dataframe = pandas.read_table('SMSSpamCollection', header=None, encoding='utf-8')
print(dataframe.info())
#print(dataframe.head(n=10))
#print(dataframe.iloc[518])

# check class distribution
smscategory = dataframe[0]
smscontent = dataframe[1]
#print(smscategory.value_counts())

# convert class labels to binary value (0 = ham, 1 = spam)
encoder = LabelEncoder()
contentLabel = encoder.fit_transform(smscategory)
#print('First 10 Category Labels: %s' % (contentLabel[:10]))

# pre-process smscontent data
## use regex to replace distinct attributes
### (emailaddr = Email Address in text)
contentProcessed = smscontent.str.replace('[^ ]+@[a-z]+\.[a-z]+(\.[a-z]+)*', 'emailaddr')

### (webaddr = http Web Address in text)
contentProcessed = contentProcessed.str.replace('(http://|https://)[^ ]+', 'webaddr')

### (moneysymb = Money Symbols in text)
contentProcessed = contentProcessed.str.replace('£|\$|₹', 'moneysymb')

### (phonenumbr = Phone Numbers in text)
contentProcessed = contentProcessed.str.replace('(\+\d{12}|\d{10})', 'phonenumbr')

### (othernumbr = Numbers in text)
contentProcessed = contentProcessed.str.replace('\d+(\.\d+)?', 'othernumbr')

## remove punctuation marks
contentProcessed = contentProcessed.str.replace('[^\w\d\s]', ' ')

## replace multiple whiatespace with single whitespace
contentProcessed = contentProcessed.str.replace('\s+', ' ')

## remove leading and trailing whaitespace
contentProcessed = contentProcessed.str.strip()

## change to lowercase
contentProcessed = contentProcessed.str.lower()

## function to wordnet pos from nltk pos tags 
def getWordnetPOSWithToken(sentence):
    dict_wordnetpostags = {'J': wordnet.ADJ,
    'N': wordnet.NOUN,
    'V': wordnet.VERB,
    'R': wordnet.ADV
    }
    result = []
    tokenpostags = nltk.pos_tag(nltk.word_tokenize(sentence))
    for postag in tokenpostags:
        wordnetpostag = dict_wordnetpostags.get(postag[1][0], wordnet.NOUN)
        result.append((postag[0], wordnetpostag))
    return result

## apply word lemmatization
lemmatizer = WordNetLemmatizer()
contentProcessed = contentProcessed.apply(lambda x: ' '.join(lemmatizer.lemmatize(elem[0], pos=elem[1]) for elem in getWordnetPOSWithToken(x)))

## remove stop words
set_stopwords = set(stopwords.words('english'))
set_stopwords.update(['u', 'urs', 'u\'ve', 'urself', 'u\'d', 'u\'ll', 'urselves', 'ur', 'u\'re'])
contentProcessed = contentProcessed.apply(lambda x: ' '.join(token for token in x.split() if token not in set_stopwords))

## apply word stemming
#stemmer = nltk.PorterStemmer()
#contentProcessed = contentProcessed.apply(lambda x: ' '.join(stemmer.stem(token) for token in x.split()))

#print(contentProcessed[:10])
#print(contentProcessed[518])

# build bag-of-words
list_contentwords = []
for content in contentProcessed:
    words = word_tokenize(content)
    for w in words:
        list_contentwords.append(w)

# apply frequency Frequency Distribution
list_contentwords = nltk.FreqDist(list_contentwords)

#print('Number of Words from Processed Content: %s' % (len(list_contentwords)))
#print('20 Most Common Words: %s' % (list_contentwords.most_common(20)))

# use 2000 most common words as featureset
list_commonwords = list_contentwords.most_common(2000)
#list_commonwords = list(map(lambda x: x[0], list_commonwords))
lcmwText, lcmwCount = zip(*list_commonwords)
list_commonwords = list(lcmwText)

# function to run through the common words list and set true/false on message words
def buildFeatures(message):
    words = word_tokenize(message)
    result = {}
    for word in list_commonwords:
        result[word] = (word in words)
    return result

# build dataset for sklearn
tmpFeatureset = list(zip(contentProcessed, contentLabel))
seed = 1
numpy.random.seed = seed
numpy.random.shuffle(tmpFeatureset)
featureset = [(buildFeatures(content), label) for (content, label) in tmpFeatureset]
print('Feature: %s' % len(featureset))
#print(featureset[0])

# split training & testing dataset
trainingset, testingset = model_selection.train_test_split(featureset, test_size=0.25, random_state=seed)
print('Training: %s' % len(trainingset))
print('Testing: %s' % len(testingset))

# Scikit-learn Classifiers with NLTK
## define training models
names = [
    'k-nearest Neighbors',
    'Decision Tree',
    'Random Forest',
    'Logistic Regression',
    'Stochastic Gradient Descent',
    'Naive Bayes',
    'Support-vector Machine (Linear kernel-type)'
    ]
classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
    ]
models = list(zip(names, classifiers))
#print(models)

## wrap models in nltk & train using SklearnClassifier
#nltk_model = SklearnClassifier(classifiers[6])
#nltk_model.train(trainingset)
#accuracy = nltk.classify.accuracy(nltk_model, testingset) * 100
#print('%s: accuracy - %s' % (names[6], accuracy))
#for name, classifier in models:
#    nltk_model = SklearnClassifier(classifier)
#    nltk_model.train(trainingset)
#    accuracy = nltk.classify.accuracy(nltk_model, testingset) * 100
#    print('%s: accuracy - %s' % (name, accuracy))

## wrap models using ensemble method & train using SklearnClassifier
nltk_ensemble = SklearnClassifier(VotingClassifier(estimators=models, voting='hard', n_jobs=-1))
nltk_ensemble.train(trainingset)
accuracy = nltk.classify.accuracy(nltk_ensemble, testingset) * 100
print('Hard Voting Ensemble: accuracy - %s' % (accuracy))

# build class level prediction of testing set, classification report, confusion matrix
testContents, testLabels = zip(*testingset)
testContents = list(testContents)
testLabels = list(testLabels)
#prediction = nltk_model.classify_many(testContents)
prediction = nltk_ensemble.classify_many(testContents)

print(classification_report(testLabels, prediction))
actual_to_predicted = pandas.DataFrame(
    confusion_matrix(testLabels, prediction),
    index = [['actual', 'actual'], ['ham', 'spam']],
    columns = [['predicted', 'predicted'], ['ham', 'spam']]
    )
print(actual_to_predicted.head())

# save the trained nltk model as a pickle in SMSSpamFilter_model.pkl
#joblib.dump(nltk_model, 'SMSSpamFilter_model.pkl')
joblib.dump(nltk_ensemble, 'SMSSpamFilter_model_01.pkl')