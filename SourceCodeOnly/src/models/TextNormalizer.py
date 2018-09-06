import nltk
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='english',use_custom_sw=1):
        self.stopwords  = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.use_sw = use_custom_sw
        self.custom_stopwords = custom_stopwords = ['year','would','make','also','new','one','us','take','go'
                                                    ,'us','last','time','get','could','use','add','tell','month','back'
                                                    'see','want','include','week','good','give','like','next','let','say']
        
    def is_punct(self, token):
        return all(unicodedata.category(char).startswith('P') for char in token)

    def is_stopword(self, token):
        if self.use_sw == 1:
            return token.lower() in self.stopwords or token.lower() in self.custom_stopwords
        else:
            return token.lower() in self.stopwords

    
    def normalize(self, allDocuments):
        docsAsList = list()
        for catDocs in allDocuments:
            for document in catDocs:
                wordsList = list()
                for paragraph in document:
                    for sentence in paragraph:
                        for token, tag in sentence:
                            if not self.is_punct(token) and not self.is_stopword(token) and not token.isdigit() and len(token) > 1:
                                wordsList.append(self.lemmatize(token, tag).lower())
                docsAsList.append(" ".join(wordsList))
        return docsAsList
    
    def lemmatize(self, token, pos_tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0], wn.NOUN)
        return self.lemmatizer.lemmatize(token, tag)

    def fit(self,X,y):
        return self

    def transform(self, X):
        return self.normalize(X)
