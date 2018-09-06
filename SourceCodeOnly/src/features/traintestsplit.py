import os
import shutil
import logging
import time
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.features import PickledCorpusReader as pcr
from src.features import CorpusLoader as cl

def get_subdirs(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def main():
    try:
        logger = logging.getLogger(__name__)
        load_dotenv(find_dotenv())
        DOC_PATTERN = os.getenv('doc_pkl_pattern')
        CAT_PATTERN = os.getenv('cat_pattern')
        # useful for finding various files
        project_dir = Path(__file__).resolve().parents[2]
        process_path = os.path.join(project_dir,'data','processed')
        train_path = os.path.join(process_path,'train')
        if not os.path.exists(train_path):
            os.makedirs(train_path)        
        test_path = os.path.join(process_path,'test')
        if not os.path.exists(test_path):
            os.makedirs(test_path)         
        processed_corpus = pcr.PickledCorpusReader(process_path,DOC_PATTERN, cat_pattern=CAT_PATTERN)
        loader = cl.CorpusLoader(processed_corpus)
        zippedXy = zip(loader.reader.abspaths(loader.X_train),loader.y_train)
        for X,y in zippedXy:
            src = os.path.abspath(X)
            dest_dir = os.path.join(train_path,y)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            file_name = os.path.basename(X)
            dest_file_path = os.path.join(dest_dir,file_name)
            shutil.move(src,dest_file_path)
        folder_paths = [dir_path for dir_path in get_subdirs(process_path) if not ((dir_path == train_path) or (dir_path == test_path))]
        for folder_path in folder_paths:
            dest_path = os.path.join(test_path,os.path.basename(folder_path))
            shutil.move(folder_path,dest_path)
    except Exception as e:
        logger.error("Error: {}".format(e))
