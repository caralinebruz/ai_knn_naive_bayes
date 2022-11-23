''' K-Nearest Neighbors Classifier
'''

# for average per row
import numpy as np
import pandas as pd


class KNN:

	def __init__(self, verbose, K):
		self.verbose = verbose
		self.K = K

	def train(self, data):
		# create a dataframe with the data
		df = pd.Dataframe(data=data)