''' K-Nearest Neighbors Classifier
'''

# for average per row
import numpy as np
import pandas as pd


class KNN:

	def __init__(self, verbose, K, distance_function):
		self.verbose = verbose
		self.K = K
		self.distance_function = distance_function

	def train(self, data):
		# create a dataframe with the data
		df = pd.DataFrame(data=data)