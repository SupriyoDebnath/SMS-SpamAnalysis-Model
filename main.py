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

from nltk.classify.scikitlearn import SklearnClassifier

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


# load dataset
dataframe = pandas.read_table('SMSSpamCollection', header=None, encoding='utf-8')
print(dataframe.info())
#print(dataframe.head(n=10))
#print(dataframe.iloc[518])

# check class distribution
smscategory = dataframe[0]
smscontent = dataframe[1]
print(smscategory.value_counts())

# convert class labels to binary value (0 = ham, 1 = spam)
encoder = LabelEncoder()
contentLabel = encoder.fit_transform(smscategory)
print('Label Volume: %s' % len(contentLabel))
#print('First 10 Category Labels: %s' % (contentLabel[:10]))

# normalize smscontent data
normalizer = TextNormalizer(smscontent)
normalizedContent = normalizer.normalize()
#print(normalizedContent[:10])
#print(normalizedContent[518])

# vectorize smscontent data
vectorizer = TextVectorizer(normalizedContent, True)
vectorizer.buildVocabulary(2000)

# build featureset for training
tmpFeatureset = list(zip(normalizedContent, contentLabel))
seed = 1
numpy.random.seed = seed
numpy.random.shuffle(tmpFeatureset)
featureset = [(vectorizer.transform(content), label) for (content, label) in tmpFeatureset]
print('Feature Volume: %s' % len(featureset))
#print('First 10 Category Labels: %s' % featureset[:10])

# split training & testing dataset
trainingset, testingset = model_selection.train_test_split(featureset, test_size=0.25, random_state=seed)
print('Training Volume: %s' % len(trainingset))
print('Testing Volume: %s' % len(testingset))

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

## wrap models using ensemble method & train using SklearnClassifier/Voting Classifier
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
joblib.dump(nltk_ensemble, 'SMSSpamFilter_model.pkl')