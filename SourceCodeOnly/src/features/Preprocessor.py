import os
import pickle
import multiprocessing as mp
from collections import defaultdict
from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

class Preprocessor(object):
    
    def __init__(self, corpus, target=None,**kwargs):
        self.corpus = corpus
        self.target = target
    results = []    
    def fileids(self, fileids=None, categories=None):
        fileids = self.corpus.resolve(fileids, categories)
        if fileids:
            return fileids
        return self.corpus.fileids()
    
    def on_error(self,error_msg):
        print(error_msg)
    
    def on_result(self, result):
        self.results.append(result)
    
    def abspath(self, fileid):
        # Find the directory, relative to the corpus root.
        parent = os.path.relpath(
            os.path.dirname(self.corpus.abspath(fileid)), self.corpus.root
        )

        # Compute the name parts to reconstruct
        basename  = os.path.basename(fileid)
        name, ext = os.path.splitext(basename)

        # Create the pickle file extension
        basename  = name + '.pickle'

        # Return the path to the file relative to the target.
        return os.path.normpath(os.path.join(self.target, parent, basename))
    
    def process(self, fileid):
        """For a single file, checks the location on disk to ensure no errors,
        uses +tokenize()+ to perform the preprocessing, and writes transformed
        document as a pickle to target location.
        """
        # Compute the outpath to write the file to.
        target = self.abspath(fileid)
        parent = os.path.dirname(target)

        # Make sure the directory exists
        if not os.path.exists(parent):
            os.makedirs(parent)

        # Make sure that the parent is a directory and not a file
        if not os.path.isdir(parent):
            raise ValueError(
                "Please supply a directory to write preprocessed data to."
            )
        # Create a data structure for the pickle
        document = list(self.tokenize(fileid))
        # Open and serialize the pickle to disk
        with open(target, 'wb') as f:
            pickle.dump(document, f, pickle.HIGHEST_PROTOCOL)
        # Clean up the document
        del document
        return target
        
    def transform(self, fileids=None, categories=None,tasks=None):
        # Reset the results
        results = []

        # Make the target directory if it doesn't already exist
        if not os.path.exists(self.target):
            os.makedirs(self.target)

        # Create a multiprocessing pool
        tasks = tasks or mp.cpu_count()
        pool  = mp.Pool(processes=tasks)

        # Enqueue tasks on the multiprocessing pool and join
        for fileid in self.fileids():
            pool.apply_async(self.process, (fileid,), callback=self.on_result,error_callback=self.on_error)

        # Close the pool and join
        pool.close()
        pool.join()
        
        return results
            
    def tokenize(self, fileid):
        for paragraph in self.corpus.paras(fileids=fileid):
            yield [
                pos_tag(sent) 
                for sent in paragraph
            ]
