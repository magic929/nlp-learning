import pickle
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier

print("Loading dataset...")

raw_text_dataset = pickle.load(open("data/raw_text_dataset.pickle", "rb"))
X_train_raw = raw_text_dataset[0]
y_train_labels = raw_text_dataset[1]
X_test_raw = raw_text_dataset[2]
y_test_labels = raw_text_dataset[3]

y_train = ["acq" in y for y in y_train_labels]
y_test = ["acq" in y for y in y_test_labels]

print(" %d training examples (%d positive)" % (len(y_train), sum(y_train)))
print(" %d test examples (%d positive)" % (len(y_test), sum(y_test)))

vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=2, stop_words='english', use_idf=True)

X_train_tfidf = vectorizer.fit_transform(X_train_raw)

print(" Actual number of tfidf features: %d" % X_train_tfidf.get_shape()[1])
t0 = time.time()

svd = TruncatedSVD(100)
lsa = make_pipeline(svd, Normalizer(copy=False))

X_train_lsa = lsa.fit_transform(X_train_tfidf)
print("sample tfidf vector: ", X_train_tfidf[0])
print("sample lsa vector: ", X_train_lsa[0])

print(" done in %.3fsec" % (time.time() - t0))

explained_variance = svd.explained_variance_ratio_.sum()
print(" Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

X_test_tfidf = vectorizer.transform(X_test_raw)
X_test_lsa = lsa.transform(X_test_tfidf)

print("\nclassifying tfidf vectors...")

t0 = time.time()

knn_tfidf = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine')
knn_tfidf.fit(X_train_tfidf, y_train)

p = knn_tfidf.predict(X_test_tfidf)

numRight = 0
for i in range(0, len(p)):
    if p[i] == y_test[i]:
        numRight += 1

print(" (%d / %d) correct - %.2f%%" % (numRight, len(y_test), float(numRight) / float(len(y_test)) * 100.0))

elpased = (time.time() - t0)
print(" done in %.3fsec" % elpased)

print("\nClassifying LSA vectors...")

t0 = time.time()

knn_lsa = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine')
knn_lsa.fit(X_train_lsa, y_train)

numRight = 0
for i in range(0, len(p)):
    if p[i] == y_test[i]:
        numRight += 1

print(" (%d / %d) correct - %.2f%%" % (numRight, len(y_test), float(numRight) / float(len(y_test)) * 100.0))

elapsed = (time.time() - t0)
print("   done in %.3fsec" % elpased)