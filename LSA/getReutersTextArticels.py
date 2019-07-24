from glob import glob
import itertools
import os.path
import re
import tarfile

import numpy as np

from html.parser import HTMLParser
from urllib.request import urlretrieve
from sklearn.datasets import get_data_home

import pickle

def _not_in_sphinx():
    return '__file__' in globals()

class ReutersParser(HTMLParser):
    def __init__(self, encoding='latin-1'):
        HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding
    
    def handle_starttag(self, tag, attrs):
        method = 'start_' + tag
        getattr(self, method, lambda x: None)(attrs)
    
    def handle_endtag(self, tag):
        method = 'end_' + tag
        getattr(self, method, lambda: None)()
    
    def _reset(self):
        self.in_title = 0
        self.in_body = 0
        self.in_topics = 0
        self.in_topic_d = 0
        self.title = ""
        self.body = ""
        self.topics = []
        self.topic_d = ""
    
    def parser(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
            self.close()
    
    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_title:
            self.title += data
        elif self.in_topic_d:
            self.topic_d += data
    
    def start_reuters(self, attributes):
        pass

    def end_reuters(self):
        self.body = re.sub(r'\s+', r'', self.body)
        self.docs.append({'title': self.title, 'body': self.body, 'topics': self.topics})
        self._reset()
    
    def start_title(self, attributes):
        self.in_title = 1
    
    def end_title(self):
        self.in_title = 0

    def start_body(self, attributes):
        self.in_body = 1
    
    def end_body(self):
        self.in_body = 0
    
    def start_topics(self, attributes):
        self.in_topics = 1
    
    def end_topics(self):
        self.in_topics = 0
    
    def start_d(self, attributes):
        self.in_topic_d = 1
    
    def end_d(self):
        self.in_topic_d = 0
        self.topics.append(self.topic_d)
        self.topic_d = ""


def stream_reuters_documents(data_path=None):
    DOWNLOAD_URL = ('http://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz')
    ARCHIVE_FILENAME = 'reuters21578.tar.gz'

    if data_path is None:
        data_path = os.path.join(get_data_home(), "reuters")
    if not os.path.exists(data_path):
        print("downloading dataset (once and for all) into %s" % data_path)
        os.mkdir(data_path)

        def progress(blocknum, bs, size):
            total_sz_mb = '%.2f MB' % (size / 1e6)
            current_sz_mb = '%.2f MB' % ((blocknum * bs) / 1e6)
            if _not_in_sphinx():
                print('\rdownloaded %s / %s' % (current_sz_mb, total_sz_mb), end='')
        
        archive_path = os.path.join(data_path, ARCHIVE_FILENAME)
        urlretrieve(DOWNLOAD_URL, filename=archive_path, reporthook=progress)

        if _not_in_sphinx():
            print('\r', end='')
        print("untarring Reuters dataset...")
        tarfile.open(archive_path, 'r:gz').extractall(data_path)
        print("done.")
    
    parser = ReutersParser()
    for filename in glob(os.path.join(data_path, "*.sgm")):
        for doc in parser.parser(open(filename, 'rb')):
            yield doc


def get_minibatch(doc_iter, size, pos_class):
    data = [(u'{title}\n\n{body}'.format(**doc), doc['topics']) for doc in itertools.islice(doc_iter, size) if doc['topics']]
    if not len(data):
        return np.asarray([], dtype=int), np.asarray([], dtype=int).tolist()
    
    X_test, y = zip(*data)

    X_test = list(X_test)
    y = list(y)

    toRemove = []
    docNum = 0

    for article in X_test:
        if article.isspace() or (article == ""):
            toRemove.append(docNum)
            
        docNum += 1
    
    toRemove.reverse()
    for i in toRemove:
        del X_test[i]
        del y[i]

    return X_test, y


def iter_minibatches(doc_iter, minibatch_size):
    X_test, y = get_minibatch(doc_iter, minibatch_size)
    while len(X_test):
        yield X_test, y
        X_test, y = get_minibatch(doc_iter, minibatch_size)


data_stream = stream_reuters_documents()

positive_class = 'acq'

X_train_raw, y_train_raw = get_minibatch(data_stream, 5000, positive_class)
X_test_raw, y_test_raw = get_minibatch(data_stream, 5000, positive_class)

print("Train set is %d documents" % (len(y_train_raw)))
print("Test set is %d documents" % (len(y_test_raw)))

pickle.dump((X_train_raw, y_train_raw, X_test_raw, y_test_raw), open("DATA/raw_text_dataset.pickle", "wb"))