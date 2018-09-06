import pickle
from src.data import BBCNewsCorpusReader as cr
class PickledCorpusReader(cr.BBCNewsCorpusReader):
    def docs(self, fileids=None, categories=None):
        fileids = self.resolve(fileids, categories)
        # Load one pickled document into memory at a time.
        for path in self.abspaths(fileids):
            with open(path, 'rb') as f:
                yield pickle.load(f)
