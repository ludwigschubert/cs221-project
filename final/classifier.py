#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Above header to deal with any non-ascii html

import sys

import numpy as np
import sklearn.metrics

from db import DB
from features import *

DEBUG_VERBOSITY = 0
DEFAULT_TRAINING_NUM = 500
DEFAULT_TESTING_NUM	= 50

class Regression(object):

	def __init__(self, numberOfTrainingSamples, numberOfTestSamples):
		self.numberOfTrainingSamples = numberOfTrainingSamples
		self.numberOfTestSamples = numberOfTestSamples
		# Initialize feature extractors
		featureExtractorClasses = [
			LengthFeatureExtractor,
			NGramFeatureExtractor
		]
		self.featureExtractors = [extractorClass() for extractorClass in featureExtractorClasses]
		# Initialize DB connection
		self.db = DB()

	def run(self):
		trainExamples, testExamples = self.db.loadRandomSamples(self.numberOfTrainingSamples, self.numberOfTestSamples)

		if (len(trainExamples) + len(testExamples) != self.numberOfTestSamples + self.numberOfTrainingSamples):
			print('\nDid not get back the expected number of database rows! \
					\nInstead, returning {} training exs and {} testing exs'.format(len(trainExamples), len(testExamples)))

		if (DEBUG_VERBOSITY > 1):
			print('Training on {} readmes, then testing on {}'.format(self.numberOfTrainingSamples, self.numberOfTestSamples))

		trainExamples  = [tuple(value for value in repo) for repo in trainExamples]
		testExamples   = [tuple(value for value in repo) for repo in testExamples]
		trainingHTML   = [trainExample[1] for trainExample in trainExamples]
		trainingScores = [trainExample[2] for trainExample in trainExamples]
		testHTML       = [testExample[1] for testExample in testExamples]
		testScores     = [testExample[2] for testExample in testExamples]

		def featureExtraction(samples, isTraining):
			featureListsPerSample = []
			for featureExtractor in self.featureExtractors:
				features = featureExtractor.extract(samples, isTraining)
				featureListsPerSample.append(features)
			featuresOfSamples = np.concatenate(featureListsPerSample, axis=1)
			return featuresOfSamples

		# Train the model
		model = linear_model.LogisticRegression(penalty = 'l2')
		trainingFeatures = featureExtraction(trainingHTML, isTraining = True)
		model.fit(trainingFeatures, trainingScores)

		# Calculate training error
		trainingPrediction = model.predict(trainingFeatures)
		trainingError = sklearn.metrics.mean_absolute_error(trainingScores, trainingPrediction)
		print "Training Error (mean absolute): {error}".format(error = trainingError)

		# Calculate test error
		testFeatures = featureExtraction(testHTML, isTraining = False)
		testPrediction = model.predict(testFeatures)
		testError = sklearn.metrics.mean_absolute_error(testScores, testPrediction)
		print "Test Error (mean absolute): {error}".format(error = testError)




def main(argv=None):
	'''
	Runs a mega-super-awesome classifier
	'''
	if argv is None:
		argv = sys.argv

	n_training_samples = DEFAULT_TRAINING_NUM
	n_testing_samples  = DEFAULT_TESTING_NUM

	if len(argv) >= 3:
		n_training_samples = int(argv[1])
		n_testing_samples	= int(argv[2])
	else:
		print('\nUsing default number for train ({}), test ({})'.format(n_training_samples, n_testing_samples))

	regression = Regression(n_training_samples, n_testing_samples)
	regression.run()

if __name__ == "__main__":
	sys.exit(main())

