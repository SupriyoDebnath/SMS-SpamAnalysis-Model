import joblib

# load the nltk model from SMSSpamFilter_model.pkl
model_smsspamfilter = joblib.load('SMSSpamFilter_model.pkl')

# predict 0 - ham, 1 - spam
## Important! input may need to be pre-processed to match with trainingset 
smscontent01 = 'Hi, how are you?'
featureset = {x:True for x in smscontent01.split()};
#print(featureset)
prediction = model_smsspamfilter.classify(featureset)
print(prediction)