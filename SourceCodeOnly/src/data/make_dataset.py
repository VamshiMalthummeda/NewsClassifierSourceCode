# -*- coding: utf-8 -*-
import logging
import os
import zipfile
import requests
import time
import shutil
import random
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.data import BBCNewsCorpusReader as cr

def DownloadFile(dir_path):
    try:
        # find .env automagically by walking up directories until it's found, then
        # load up the .env entries as environment variables
        load_dotenv(find_dotenv())
        url = os.getenv('url')
        file_name = url[url.rfind("/",0) + 1:]
        logger = logging.getLogger(__name__)
        resp = requests.get(url)
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'wb') as zfile:
            zfile.write(resp.content)
            zfile.close()
        logger.info("Download Process Completed Successfully")
    except Exception as e:
        logger.info("Error: {}".format(e))
    return file_name

def get_subdirs(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def mainProcess(input_filepath,output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    load_dotenv(find_dotenv())
    dirname = os.getenv('initialdir')
    logger = logging.getLogger(__name__)
    with zipfile.ZipFile(input_filepath,"r") as zip_ref:
        zip_ref.extractall(output_filepath)
    logger.info("Extraction Process Completed Successfully")
    initialdir = os.path.join(output_filepath,dirname)
    labeleddir = os.path.join(output_filepath,'labeled')
    unlabeled_dir = os.path.join(output_filepath,'unlabeled')
    min_digit = int(os.getenv('mindigit'))
    file_ext = os.getenv('fileext')
    if os.path.exists(labeleddir):
        shutil.rmtree(labeleddir)
    if not os.path.exists(unlabeled_dir):
        os.makedirs(unlabeled_dir)
    else:
        shutil.rmtree(unlabeled_dir)
        os.makedirs(unlabeled_dir)
    for bbc_dir in get_subdirs(initialdir):
        for ctr in range(0,min_digit):
            file_name = str(random.randint(1,400)).zfill(min_digit) + file_ext
            if os.path.exists(os.path.join(bbc_dir,file_name)):
                dest_path = os.path.join(unlabeled_dir,file_name)
                src_path = os.path.join(bbc_dir,file_name)
                shutil.move(src_path,dest_path)  
    os.rename(initialdir,labeleddir)
    logger.info('making final data set from raw data')

def GenerateAnalytics(dirpath):
    logger = logging.getLogger(__name__)
    load_dotenv(find_dotenv())
    DOC_PATTERN = os.getenv('doc_pattern')
    CAT_PATTERN = os.getenv('cat_pattern')
    ENCODING = os.getenv('encoding')
    
    corpus_path = os.path.join(dirpath,'data','raw','labeled')
    corpus = cr.BBCNewsCorpusReader(corpus_path,DOC_PATTERN, cat_pattern=CAT_PATTERN,encoding=ENCODING)
    logging.info("Starting: Corpus Analytics Retrieval")
    x = corpus.describe()
    logger.info(x)
    logger.info("END: Corpus Analytics Retrieval")

def main():
    LOG_NAME = "process.log"
    # useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    ext_path = os.path.join(project_dir,'data','external')
    raw_path = os.path.join(project_dir,'data','raw')
    log_path = os.path.join(project_dir,LOG_NAME)
    log_fmt = '%(processName)-10s %(module)s %(asctime)s %(message)s'
    logging.basicConfig(filename=log_path,level=logging.INFO, format=log_fmt)
    zip_file = DownloadFile(ext_path)
    input_path = os.path.join(ext_path,zip_file)
    mainProcess(input_path,raw_path)
    GenerateAnalytics(project_dir)
