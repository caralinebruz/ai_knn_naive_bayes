''' Naive Bayes Classifier
'''

# for average per row
import numpy as np
import pandas as pd


class NaiveBayes:

	def __init__(self, verbose, correction):
		self.verbose = verbose
		self.correction = correction


	def train(self, data):
		pass