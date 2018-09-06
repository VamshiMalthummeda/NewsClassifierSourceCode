import os
import pickle
import logging
import time
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import multiprocessing as mp
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
    logger.info(error_msg)

def generate_models(project_dir):
    try:
        logger = logging.getLogger(__name__)
        load_dotenv(find_dotenv())
        DOC_PATTERN = os.getenv('doc_pkl_pattern')
        CAT_PATTERN = os.getenv('cat_pattern')
        PROCESS_DIR_NAME = 'processed'
        process_path = os.path.join(project_dir,'data','processed')
        test = os.path.join(process_path,'test')
        train = os.path.join(process_path,'train')
        models_path = os.path.join(project_dir,'models')
        test_corpus = pcr.PickledCorpusReader(test,DOC_PATTERN, cat_pattern=CAT_PATTERN)
        X_test_data = [list([doc]) for doc in test_corpus.docs()]
        y_test_data = [test_corpus.categories(fileids=[fileid])[0] for fileid in test_corpus.fileids()]
        trained_corpus = pcr.PickledCorpusReader(train,DOC_PATTERN, cat_pattern=CAT_PATTERN)
        X_train_data = [list([doc]) for doc in trained_corpus.docs()]
        y_train_data = [trained_corpus.categories(fileids=[fileid])[0] for fileid in trained_corpus.fileids()]
        category_data = trained_corpus.categories()
        namesdict = {0:"MultinomialNB",1:"LogisticRegression",2:"LinearSVC",3:"SGDClassifier"}
        models = []
        for form in (MultinomialNB,LogisticRegression,LinearSVC,SGDClassifier):
            models.append(create_pipeline(form()))
        # Make sure the models directory exists
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        ctr = 0
        for model in models:
            logger.info("Starting:GridSearchCV for {} model".format(namesdict[ctr]))
            search = GridSearchCV(model, cv=12,param_grid={
                'vectorize__max_df' : [0.5],
                'vectorize__min_df' : [0.1],
                'vectorize__smooth_idf' : [True],
                'vectorize__norm' : ['l2'],
                'vectorize__ngram_range':[(1,1),(1,2)],
                'vectorize__sublinear_tf': [True]
                },n_jobs=-1)
            search.fit(X_train_data,y_train_data)
            logger.info("The best score for {} model is {}".format(namesdict[ctr],search.best_score_))
            logger.info("The best params for {} model is {}".format(namesdict[ctr],search.best_params_))
            logger.info("END:GridSearchCV for {} model".format(namesdict[ctr]))
            
            path = os.path.join(models_path,'{}.pickle'.format(namesdict[ctr]))
            with open(path, 'wb') as f:
                pickle.dump(search, f,pickle.HIGHEST_PROTOCOL)
            logger.info("SAVED:{} model".format(namesdict[ctr]))
            ctr+=1

    except Exception as e:
        logger.info("Error: {}".format(e))

def main():
    LOG_NAME = "process.log"
    project_dir = Path(__file__).resolve().parents[2]
    log_path = os.path.join(project_dir,LOG_NAME)
    log_fmt = '%(processName)-10s %(module)s %(asctime)s %(message)s'
    logging.basicConfig(filename=log_path,level=logging.INFO, format=log_fmt)     
    generate_models(project_dir)
    

