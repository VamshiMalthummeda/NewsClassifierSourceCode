import os
import pickle
import logging
from statistics import mode
from pathlib import Path
import multiprocessing as mp
from nltk.corpus.reader import PlaintextCorpusReader
from nltk import pos_tag, sent_tokenize, wordpunct_tokenize
from flask import Flask, jsonify,request,render_template
app = Flask(__name__)

class VoteClassifier(PlaintextCorpusReader):
    def __init__(self,root,model_path,fileids=None,**kwargs):
        super(VoteClassifier, self).__init__(root,fileids)
        self._results = list()
        self._classifiers = []
        self._models_path = model_path
         
    def get_classifier(self,model_path):
        classifier_f = open(model_path,"rb")
        classifier = pickle.load(classifier_f)
        classifier_f.close()
        return classifier
    def on_result(self,result):
        self._classifiers.append(result)
    def on_error(error_msg):
        self._message = error_msg
    def load_classifiers(self):
        for name in os.listdir(self._models_path):
            fpath = os.path.join(self._models_path,name)
            self._classifiers.append(self.get_classifier(fpath))
    def predict(self, features):
        votes = []
        for c in self._classifiers:
            v = c.predict(features)
            votes.append(v[0])
        
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.predict(features)
            votes.append(v[0])

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf    
    def preprocessDoc(self,fileid):
        return [[list(self.tokenize(fileid))]]
        
    def tokenize(self, fileid):
        for paragraph in self.paras(fileids=fileid):
            yield [
                pos_tag(sent) 
                for sent in paragraph
            ]        
        
        

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/news_type/<string:fileid>')
def predict_news_type(fileid):
    LOG_NAME = "process.log"
    project_dir = Path(app.root_path).resolve().parents[1]
    log_path = os.path.join(project_dir,LOG_NAME)
    log_fmt = '%(processName)-10s %(module)s %(asctime)s %(message)s'
    logging.basicConfig(filename=log_path,level=logging.INFO, format=log_fmt)
    voteClassifier = VoteClassifier(os.path.join(project_dir,'data','raw','unlabeled'),os.path.join(project_dir,'models'))
    logging.info("Instantiated Successfully")
    voteClassifier.load_classifiers()
    logging.info("loaded classifiers Successfully")
    features = voteClassifier.preprocessDoc(fileid)
    logging.info("extracted features Successfully")
    predictVal = voteClassifier.predict(features)
    confVal = voteClassifier.confidence(features)
    return jsonify({'prediction':predictVal,'confidence':confVal})
    
app.run(port=5000)
