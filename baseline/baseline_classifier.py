#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Above header to deal with any non-ascii html

import random
import collections
import math
import sys
import copy
from collections import Counter
from util import *
import psycopg2

DEBUG_VERBOSITY = 2
DEFAULT_TRAINING_NUM = 10
DEFAULT_TESTING_NUM	= 10

ETA = 1
NUM_ITERS = 20

def extractWordFeatures(x):
	"""
	Taken from sentiment homework.

	Extract word features for a string x. Words are delimited by
	whitespace characters only.
	@param string x: 
	@return dict: feature vector representation of x.
	Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
	"""
	retval = collections.defaultdict(lambda: 0.0)
	for word in x.split():
		retval[word] = retval[word] + 1.0
	return retval
	
def extractLengthFeature(x):
	length = len(x)
	log = math.log(length**2, 10)
	key = 'order_of_length_{0:g}'.format( math.floor(log))
	return {key: 1}

def learnPredictor(trainExamples, testExamples, featureExtractor):
	'''
	Taken from sentiment homework.

	Given |trainExamples| and |testExamples| (each one is a list of (x,y)
	pairs), a |featureExtractor| to apply to x, and the number of iterations to
	train |numIters|, return the weight vector (sparse feature vector) learned.

	You should implement stochastic gradient descent.

	Note: only use the trainExamples for training!
	You should call evaluatePredictor() on both trainExamples and testExamples
	to see how you're doing as you learn after each iteration.
	numIters refers to a variable you need to declare. It is not passed in.
	'''
	# weights = {}	# feature => weight
	# eta = ETA
	# numIters = NUM_ITERS
	# def predictor(string):
	# 	dp = dotProduct(featureExtractor(string),weights)
	# 	if (dp > 0):
	# 		return 1
	# 	else:
	# 		return -1
	# for i in range(1,numIters+1):
	# 	for _, string, y in trainExamples:
	# 		phi = featureExtractor(string);
	# 		if (dotProduct(weights, phi)*y < 1):
	# 			grad = {}
	# 			increment(grad, -1, phi)
	# 			increment(weights, -1*y*eta, grad)
	# 	
	# 	print 'Weights %s', Counter(weights).most_common()
	# 	print 'Training error on iteration {0:g}: {1:f}'.format(i, evaluatePredictor(trainExamples, predictor))
	# 	print 'Testing error on iteration {0:g}:	{1:f}'.format(i, evaluatePredictor(testExamples,	predictor))
	# return weights
	weights = {}	# feature => weight
	# BEGIN_YOUR_CODE (around 15 lines of code expected)
	numIters = 15
	stepSize = 0.001
	debug = False
	
	def score(example, label):
		features = featureExtractor(example)
		return dotProduct(features, weights)
	
	def residual(example, label):
		return score(example, label) - label
	
	def gradientSquaredLoss(example, label, weights):
		return { key: (value * 2 * residual(example, label)) for key, value in featureExtractor(example).iteritems() }
	
	def predict(example):
		return score(example, label)
	
	for i in range(numIters):
		for github_id, example, label in trainExamples:
			loss = gradientSquaredLoss(example, label, weights)
			increment(weights, -stepSize, loss)
			print github_id
			print loss
			print score(example, label)
		print("====")
		# print("Iteration: %s" % i)
		print("Train Examples: %s" % evaluatePredictor(trainExamples, predict))
		print("Test Examples: %s" % evaluatePredictor(testExamples, predict))
		print 'Weights %s', Counter(weights).most_common()[0:10]
		# print 'Training error on iteration {0:g}:	{1:f}'.format(i, evaluatePredictor(trainExamples, predictor))
		# print 'Testing error on iteration {0:g}:	{1:f}'.format(i, evaluatePredictor(testExamples,	predictor))
	
	# END_YOUR_CODE
	return weights

def getReadmes(num):
	'''
	Make connection to postgres server and get num readme entries

	We want to return a list of lists [[id, readme_text, stars, ...], [id, readme_text, stars, ...], ...]
	'''
	try:
		conn=psycopg2.connect("dbname='data-collector_development'") # user='dbuser' password='mypass'")
	except:
		print "I am unable to connect to the database."

	cur = conn.cursor()
	try:
		#TODO: do we want a specific subset? Will this be repeatable?
		cur.execute('SELECT id, readme_html, stargazers_count FROM repositories WHERE repositories.readme_html IS NOT NULL LIMIT {}'.format(num))
	except:
		print "I can't SELECT from repositories"

	rows = cur.fetchall()

	if (DEBUG_VERBOSITY > 2):
		print "\nRows: \n"
		for row in rows:
			print "	", row[1]

	return rows # should be in correct form

def getRandomSample(ntrain, ntest):
	'''
	Get a bunch of data from db.
	Randomly pick from your data (to simulate random draw)
	Assign ntrain to training set, ntest to testing set
	'''

	training_ids = set()
	testing_ids	= set()

	training_data = []
	testing_data	= []

	data = getReadmes((ntrain+ntest)*10) # pick randomly from a pool 10x the total desired size

	while (len(training_data) < ntrain):
		testidx = random.randint(0,len(data)-1)
		if (testidx in training_ids or testidx in testing_ids):
			continue
		training_data.append(data[testidx])
		training_ids.add(testidx)

	while (len(testing_data) < ntest):
		testidx = random.randint(0,len(data)-1)
		if (testidx in training_ids or testidx in testing_ids):
			continue
		testing_data.append(data[testidx])
		testing_ids.add(testidx)
	# print("Training Data has %s items, testing data ahs %s items" %)
	return (training_data, testing_data)



def main(argv=None):
	'''
	Runs a naive baseline classifier
	'''
	if argv is None:
		argv = sys.argv

	n_training_samples = DEFAULT_TRAINING_NUM
	n_testing_samples	= DEFAULT_TESTING_NUM
	if len(argv) >= 3:
		n_training_samples = argv[2]
		n_testing_samples	= argv[3]
	else:
		print('\nUsing default number for train ({}), test ({})'.format(n_training_samples, n_testing_samples))

	readme_data = getReadmes(n_training_samples+n_testing_samples)

	trainExamples, testExamples = getRandomSample(n_training_samples, n_testing_samples)

	if (len(trainExamples) + len(testExamples) != n_testing_samples + n_training_samples):
		print('\nDid not get back the expected number of database rows! \
				\nInstead, returning {} training exs and {} testing exs'.format(len(trainExamples), len(testExamples)))

	if (DEBUG_VERBOSITY > 1):
		print('Training on {} readmes, then testing on {}'.format(n_training_samples, n_testing_samples))
	trainExamples = [tuple(value / 100000.0 if index == 2 else value for index, value in enumerate(repo)) for repo in trainExamples]
	testExamples  = [tuple(value / 100000.0 if index == 2 else value for index, value in enumerate(repo)) for repo in testExamples]
	featureExtractor = extractWordFeatures
	weights = learnPredictor(trainExamples, testExamples, featureExtractor)


if __name__ == "__main__":
	sys.exit(main())

