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

DEBUG_VERBOSITY = 3
DEFAULT_TRAINING_NUM = 100
DEFAULT_TESTING_NUM  = 100

def extractWordFeatures(x):
    """
    Taken from sentiment homework.

    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    retval = collections.defaultdict(lambda: 0)
    for word in x.split():
        retval[word] = retval[word] + 1
    return retval

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
    weights = {}  # feature => weight
    eta = .05
    numIters = 20
    useWordFeatures = False # when false, use character features
    def predictor(string):
        dp = dotProduct(featureExtractor(string),weights)
        if (dp > 0):
            return 1
        else:
            return -1
    for i in range(1,numIters+1):
        for string, y in trainExamples:
            phi = featureExtractor(string);
            if (dotProduct(weights, phi)*y < 1):
                grad = {}
                increment(grad, -1, phi)
                increment(weights, -1*y*eta, grad)
        
        print 'Training error on iteration {0:g}: {1:f}'.format(i, evaluatePredictor(trainExamples, predictor))
        print 'Testing error on iteration {0:g}:  {1:f}'.format(i, evaluatePredictor(testExamples,  predictor))
    return weights

    def getReadmes(num):
    	'''
    	Make connection to postgres server and get num readme entries

    	We want to return a list of lists [[readme_text, stars, ...], [readme_text, stars, ...], ...]
    	'''
    	try:
		    conn=psycopg2.connect("dbname='foo' user='dbuser' password='mypass'")
		except:
		    print "I am unable to connect to the database."

		cur = conn.cursor()
		try:
			#TODO: do we want a specific subset? Will this be repeatable?
		    cur.execute('SELECT readme_text, star from bar LIMIT {}'.format(num))
		except:
		    print "I can't SELECT from bar"

		rows = cur.fetchall()

		if (DEBUG_VERBOSITY > 2):
			print "\nRows: \n"
			for row in rows:
			    print "   ", row[1]

		return rows # should be in correct form


    def main(argv=None):
    	'''
    	Runs a naive baseline classifier
    	'''
	    if argv is None:
	        argv = sys.argv

	    n_training_samples = DEFAULT_TRAINING_NUM
	    n_testing_samples  = DEFAULT_TESTING_NUM
	    if len(argv) >= 3:
	    	n_training_samples = argv[2]
	    	n_testing_samples  = argv[3]
	    else:
	    	print('\nUsing default number for train ({}), test ({})'.format(n_training_samples, n_testing_samples))

	    readme_data = getReadmes(n_training_samples+n_testing_samples)

	    if (len(readme_data) != n_testing_samples + n_training_samples):
	    	print('\nDid not get back the expected number of database rows!!')

	    trainExamples = readme_data[:n_training_samples] # get from DB
	    testExamples = readme_data[n_training_samples:] # get from DB

	    if (DEBUG_VERBOSITY > 2):
	    	print('Training on {} readmes, then testing on {}'.format(n_training_samples, n_testing_samples))

	    featureExtractor = extractWordFeatures
	    weights = learnPredictor(trainExamples, testExamples, featureExtractor)


	if __name__ == "__main__":
		sys.exit(main())