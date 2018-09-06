import os
import pickle
import logging
import time
import multiprocessing as mp
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.features import PickledCorpusReader as pcr
from src.models import TextNormalizer as tn
from src.features import CorpusLoader as cl
from sklearn.base import BaseEstimator, TransformerMixin
from yellowbrick.text.freqdist import FreqDistVisualizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import GridSearchCV
from yellowbrick.classifier import ClassificationReport,ConfusionMatrix

class ModelError(Exception):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

def create_pipeline(estimator):
    steps = [
        ('normalize', tn.TextNormalizer()),
        ('vectorize', TfidfVectorizer()),
        ('classifier',estimator)
    ]
    return Pipeline(steps)

def on_result(result):
    return result
def on_error(error_msg):
    print(error_msg)
    
def get_classifier(model_path):
    classifier_f = open(model_path,"rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier    
def generate_report(estimator,name,reports_path,counter,*data):
    categories = data[0]
    X_train_data = data[1]
    y_train_data = data[2]
    X_test_data = data[3]
    y_test_data = data[4]
    visualizer = None
    visualizer = ClassificationReport(estimator, classes=categories)
    visualizer.fit(X_train_data, y_train_data)
    visualizer.score(X_test_data, y_test_data)
    visualizer.set_title(name + "- Classification Report")
    class_rep_path = os.path.join(reports_path,name + "-" + "Classification Report" + ".png")
    visualizer.poof(outpath=class_rep_path)
    visualizer.finalize()

def generate_freq_dist(reports_path,use_custom_sw,*data):
    X_train_data = data[0]
    y_train_data = data[1]
    text_normalizer = tn.TextNormalizer(use_custom_sw=use_custom_sw)
    corpus_data = text_normalizer.fit_transform(X_train_data,y_train_data)
    tfidf_vect = TfidfVectorizer(max_df = 0.5,min_df = 0.1,smooth_idf = True,norm='l2',ngram_range=(1,1),sublinear_tf=True)
    docs = tfidf_vect.fit_transform(corpus_data,y_train_data)
    features = tfidf_vect.get_feature_names()
    visualizer = None
    visualizer = FreqDistVisualizer(features=features)
    visualizer.fit(docs)
    freq_rep_path = ''
    if use_custom_sw == 0:
        visualizer.set_title("Frequency Distribution Before SW Removal")
        freq_rep_path = os.path.join(reports_path,"FreqDist_Before_SW_Removal.png")
    else:
        visualizer.set_title("Frequency Distribution After SW Removal")
        freq_rep_path = os.path.join(reports_path,"FreqDist_After_SW_Removal.png")        
    visualizer.poof(outpath=freq_rep_path)
    visualizer.finalize()

def generate_matrix(estimator,name,reports_path,counter,*data):
    categories = data[0]
    X_train_data = data[1]
    y_train_data = data[2]
    X_test_data = data[3]
    y_test_data = data[4]
    visualizer = None
    visualizer = ConfusionMatrix(estimator, classes=categories)
    visualizer.fit(X_train_data, y_train_data)
    visualizer.score(X_test_data, y_test_data)
    visualizer.set_title(name + "- Confusion Matrix") 
    conf_mat_path = os.path.join(reports_path,name + "-" + "Confusion Matrix" + ".png")
    visualizer.poof(outpath=conf_mat_path)
    visualizer.finalize()
    
def generate_reports(project_dir):
    try:
        
        logger = logging.getLogger(__name__)
        load_dotenv(find_dotenv())
        DOC_PATTERN = os.getenv('doc_pkl_pattern')
        CAT_PATTERN = os.getenv('cat_pattern')
        PROCESS_DIR_NAME = 'processed'
        process_path = os.path.join(project_dir,'data',PROCESS_DIR_NAME)
        test = os.path.join(process_path,'test')
        train = os.path.join(process_path,'train')
        models_path = os.path.join(project_dir,'models')
        rep_path = os.path.join(project_dir,'reports/figures')
        test_corpus = pcr.PickledCorpusReader(test,DOC_PATTERN, cat_pattern=CAT_PATTERN)
        X_test_data = [list([doc]) for doc in test_corpus.docs()]
        y_test_data = [test_corpus.categories(fileids=[fileid])[0] for fileid in test_corpus.fileids()]
        trained_corpus = pcr.PickledCorpusReader(train,DOC_PATTERN, cat_pattern=CAT_PATTERN)
        X_train_data = [list([doc]) for doc in trained_corpus.docs()]
        y_train_data = [trained_corpus.categories(fileids=[fileid])[0] for fileid in trained_corpus.fileids()]
        category_data = trained_corpus.categories()
        # Make sure the models directory exists
        if not os.path.exists(models_path):
            raise ModelError(models_path,"Models does not exist at the specified path")
        # Make sure the reports directory exists
        if not os.path.exists(rep_path):
            os.makedirs(rep_path)
        p = mp.Process(target=generate_freq_dist,args=(rep_path,0,X_train_data,y_train_data))
        p.start()
        p.join()
        p = mp.Process(target=generate_freq_dist,args=(rep_path,1,X_train_data,y_train_data))
        p.start()
        p.join() 
        ctr=0
        for name in os.listdir(models_path):
            fpath = os.path.join(models_path,name)
            classifier = get_classifier(fpath)
            clf_name,_ = name.split(".")
            p = mp.Process(target=generate_report,args=(classifier,clf_name,rep_path,ctr,category_data,X_train_data,y_train_data,X_test_data,y_test_data))
            p.start()
            p.join()
            p = mp.Process(target=generate_matrix,args=(classifier,clf_name,rep_path,ctr,category_data,X_train_data,y_train_data,X_test_data,y_test_data))
            p.start()
            p.join()
            ctr+=1
    except Exception as e:
        logger.info("Error: {}".format(e))

def main():
    LOG_NAME = "process.log"
    project_dir = Path(__file__).resolve().parents[2]
    log_path = os.path.join(project_dir,LOG_NAME)
    log_fmt = '%(processName)-10s %(module)s %(asctime)s %(message)s'
    logging.basicConfig(filename=log_path,level=logging.INFO, format=log_fmt)
    generate_reports(project_dir)
