import numpy as np
from sklearn.model_selection import train_test_split,KFold

class CorpusLoader(object):
    def __init__(self, reader, folds=12, shuffle=True, categories=None):
        self.reader = reader
        self.files  = np.asarray(self.reader.fileids(categories=categories))
        self.folds  = KFold(n_splits=folds, shuffle=shuffle)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.fileids(), self.labels(), test_size=0.33,shuffle=shuffle)
        
    def fileids(self,idx=None):
        files = self.files
        if idx is None:
            return files
        return files[idx]
    
    def documents(self,idx=None):
        for fileid in self.fileids(idx):
            yield list(self.reader.docs(fileids=[fileid]))
            
    def labels(self,idx=None):
        return [
            self.reader.categories(fileids=[fileid])[0]
            for fileid in self.fileids(idx)
        ]
    def __iter__(self):
        for train_index, test_index in self.folds.split(self.X_train):
            X_train_fold = self.documents(train_index)
            y_train_fold = self.labels(train_index)

            X_test_fold = self.documents(test_index)
            y_test_fold = self.labels(test_index)

            yield X_train_fold, X_test_fold, y_train_fold, y_test_fold
