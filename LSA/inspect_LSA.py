import pickle
import time
import numpy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from pylab import *

import random

print("Loading dataset...")

raw_text_dataset = pickle.load(open("DATA/raw_text_dataset.puckle", 'rb'))
X_train_raw = raw_text_dataset[0]

print(" %d training examples" % (len(X_train_raw)))

vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=2, stop_words='english', use_idf=True)

X_train_tfidf = vectorizer.fit_transform(X_train_raw)

print("  Actual number of tfidf features: %d" % X_train_tfidf.get_shape()[1])

feat_names = vectorizer.get_feature_names()

print("Some random words in the vocabulary: ")
for i in range(0, 10):
    featNum = random.randint(0, len(feat_names))
    print(" %s" % feat_names[featNum])

print("\nPerforming dimensionality reduction using LSA")
t0 = time.time()

svd = TruncatedSVD(100)
lsa = make_pipeline(svd, Normalizer(copy=False))

X_train_lsa = lsa.fit_transform(X_train_tfidf)

for compNum in range(0, 10):
    comp = svd.components_[compNum]

    indeces = numpy.argsort(comp).tolist()

    indeces.reverse()

    terms = [feat_names[weightIndex] for weightIndex in indeces[0:10]]
    weights = [comp[weightIndex] for weightIndex in indeces[0:10]]

    terms.reverse()
    weights.reverse()
    positions = arange(10) + .5

    figure(compNum)
    barh(positions, weights, align='center')
    yticks(positions, terms)
    title('Strongest terms for component %d' % (compNum))
    grid(True)
    show()
