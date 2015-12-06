import numpy as np
from bs4 import BeautifulSoup
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

# Class: FeatureExtractor
# ---------------------
# Abstract Feature Extractor
# You should use feature extractors to encapsulate dependencies and state
class FeatureExtractor(object):

	# The training flag can be used if your extraction depends on internal state
	# e.g. a vectorizer
	# Returns: a numpy array.
	def extract(self, samples, training = True):
		pass


class StatelessFeatureExtractor(FeatureExtractor):

	def extract(self, samples, _):
		return np.array([ [self.extractFromSingleSample(sample)] for sample in samples ])

	def extractFromSingleSample(self, sample):
		pass


class LengthFeatureExtractor(StatelessFeatureExtractor):

	def extractFromSingleSample(self, sample):
		return len(sample)


class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

class NGramFeatureExtractor(FeatureExtractor):

	def getBigramVectorizedCorpus(self, corpus):
		bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b', min_df=1)
		vec = bigram_vectorizer.fit_transform(corpus).toarray()
		return bigram_vectorizer, vec

	def getTfidfBigramVectorizedCorpus(self, corpus):
		tfidf_bigram_vectorizer = TfidfVectorizer(input='content', encoding='utf-8', decode_error='strict', \
									strip_accents=None, lowercase=True, preprocessor=None, tokenizer=LemmaTokenizer(), \
									analyzer='word', stop_words='english', token_pattern=r'\b\w+\b', ngram_range=(2, 2),
									max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, \
									norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
		vec = tfidf_bigram_vectorizer.fit_transform(corpus).toarray()
		return tfidf_bigram_vectorizer, vec

	def __init__(self):
		# Choose vectorizer here
		self.corpusVectorizer = self.getTfidfBigramVectorizedCorpus
		# self.corpusVectorizer = self.getBigramVectorizedCorpus # same as getVectorizedCorpus used for progress report
		self.vectorizer = None

	def getCorpus(self, readmeHTMLs):
		corpus = []
		for readme in readmeHTMLs:
			text = BeautifulSoup(readme, 'html.parser').get_text()
			corpus.append(text)
		return corpus

	def extract(self, samples, training):
		corpus = self.getCorpus(samples)
		assert len(corpus) == len(samples)
		if training:
			vectorizer, vectors = self.corpusVectorizer(corpus)
			self.vectorizer = vectorizer
		else:
			assert self.vectorizer
			vectors = self.vectorizer.transform(corpus).toarray()
		assert len(vectors) == len(samples)
		return vectors