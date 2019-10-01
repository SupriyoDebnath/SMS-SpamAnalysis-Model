import nltk
import joblib

from nltk.tokenize import word_tokenize

class TextVectorizer:
    normalizedMessages = ''
    vocabulary = ''

    def __init__(self, messages, toBeTrained):
        self.normalizedMessages = messages
        if not toBeTrained:
            self.vocabulary = joblib.load('SMSSpamFilter_vocabulary.pkl')
    
    def buildVocabulary(self, wordvolume):
        bagOfWords = []

        for message in self.normalizedMessages:
            words = word_tokenize(message)
            for word in words:
                bagOfWords.append(word)
        bagOfWords = nltk.FreqDist(bagOfWords)
        commonWords = bagOfWords.most_common(wordvolume)
        cw_text, cw_count = zip(*commonWords)
        self.vocabulary = list(cw_text)

        joblib.dump(self.vocabulary, 'SMSSpamFilter_vocabulary.pkl')

    def transform(self, message):
        result = {}

        words = word_tokenize(message)
        for word in self.vocabulary:
            result[word] = (word in words)
        
        return result




