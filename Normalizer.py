import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

class TextNormalizer:
    rawMessages = ''
    
    def __init__(self, messages):
        self.rawMessages = messages
    
    def normalize(self):
        normalizedMessages = ''

        #print(self.rawMessages)
        normalizedMessages = self.preprocess(self.rawMessages)
        #print(normalizedMessages)
        #normalizedMessages = self.stem(normalizedMessages)
        #print(normalizedMessages)
        normalizedMessages = self.lemmatize(normalizedMessages)
        #print(normalizedMessages)
        normalizedMessages = self.removeStopwords(normalizedMessages)
        #print(normalizedMessages)

        return normalizedMessages
    
    def preprocess(self, messages):
        preprocessedMessages = ''

        # (emailaddr = Email Address in text)
        preprocessedMessages = messages.str.replace('[^ ]+@[a-z]+\.[a-z]+(\.[a-z]+)*', 'emailaddr')

        # (webaddr = http Web Address in text)
        preprocessedMessages = preprocessedMessages.str.replace('(http://|https://)[^ ]+', 'webaddr')

        # (moneysymb = Money Symbols in text)
        preprocessedMessages = preprocessedMessages.str.replace('£|\$|₹', 'moneysymb')

        # (phonenumbr = Phone Numbers in text)
        preprocessedMessages = preprocessedMessages.str.replace('(\+\d{12}|\d{10})', 'phonenumbr')

        # (othernumbr = Numbers in text)
        preprocessedMessages = preprocessedMessages.str.replace('\d+(\.\d+)?', 'othernumbr')

        # remove punctuation marks
        preprocessedMessages = preprocessedMessages.str.replace('[^\w\d\s]', ' ')

        # replace multiple whitespace with single whitespace
        preprocessedMessages = preprocessedMessages.str.replace('\s+', ' ')

        # remove leading and trailing whaitespace
        preprocessedMessages = preprocessedMessages.str.strip()

        # change to lowercase
        preprocessedMessages = preprocessedMessages.str.lower()

        return preprocessedMessages
    
    def stem(self, messages):
        stemmedMessages = ''

        stemmer = nltk.PorterStemmer()
        stemmedMessages = messages.apply(lambda x: ' '.join(stemmer.stem(token) for token in x.split()))

        return stemmedMessages

    def getWordnetPOSWithToken(self, message):
        dict_wordnetpostags = {'J': wordnet.ADJ, 
                                'N': wordnet.NOUN, 
                                'V': wordnet.VERB, 
                                'R': wordnet.ADV 
                            }
        result = []
        
        tokenpostags = nltk.pos_tag(word_tokenize(message))
        for postag in tokenpostags:
            wordnetpostag = dict_wordnetpostags.get(postag[1][0], wordnet.NOUN)
            result.append((postag[0], wordnetpostag))
        
        return result
    
    def lemmatize(self, messages):
        lemmatizedMessages = ''

        lemmatizer = WordNetLemmatizer()
        lemmatizedMessages = messages.apply(lambda x: ' '.join(lemmatizer.lemmatize(elem[0], pos=elem[1]) for elem in self.getWordnetPOSWithToken(x)))

        return lemmatizedMessages
    
    def removeStopwords(self, messages):
        stopremovedMessages = ''

        set_stopwords = set(stopwords.words('english'))
        set_stopwords.update(['u', 'urs', 'u\'ve', 'urself', 'u\'d', 'u\'ll', 'urselves', 'ur', 'u\'re'])
        stopremovedMessages = messages.apply(lambda x: ' '.join(token for token in x.split() if token not in set_stopwords))

        return stopremovedMessages



