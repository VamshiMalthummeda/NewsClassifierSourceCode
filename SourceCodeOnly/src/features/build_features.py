import time
import logging
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.data import BBCNewsCorpusReader as cr
from src.features import Preprocessor as pp

def PreProcessCorpus(project_dir):
    try:
        logger = logging.getLogger(__name__)
        corpus_path = os.path.join(project_dir,'data','raw','labeled')
        load_dotenv(find_dotenv())
        DOC_PATTERN = os.getenv('doc_pattern')
        CAT_PATTERN = os.getenv('cat_pattern')
        ENCODING = os.getenv('encoding')
        process_path = os.path.join(project_dir,'data','processed')
        corpus = cr.BBCNewsCorpusReader(corpus_path,DOC_PATTERN, cat_pattern=CAT_PATTERN,encoding=ENCODING)
        processed_corpus = pp.Preprocessor(corpus,process_path)
        logger.info("Starting: Corpus Preprocessing")
        start = time.time()
        processed_corpus.transform()
        timediff = time.time()-start
        logger.info('The time taken by above operation is:{0:8.2f}'.format(timediff))
        logger.info("END: Corpus Preprocessing")
    except Exception as e:
        logger.error("Error: {}".format(e))
        

def main():
    LOG_NAME = "process.log"
    project_dir = Path(__file__).resolve().parents[2]
    log_path = os.path.join(project_dir,LOG_NAME)
    log_fmt = '%(processName)-10s %(module)s %(asctime)s %(message)s'
    logging.basicConfig(filename=log_path,level=logging.INFO, format=log_fmt)    
    PreProcessCorpus(project_dir)
