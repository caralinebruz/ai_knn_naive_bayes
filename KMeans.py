''' K-Means Clustering Unsupervised Learning
'''
import sys
import numpy as np
import pandas as pd

import utils
from utils import euclidean_distance_sq




class KMeans:

	def __init__(self, distance_function):
		self.distance_function = distance_function
		self.centroids = []
		self.train_df = None
		self.train_data = None
		self.numcols = 0


	def clean(self,text):
		''' Prepares a file for use. Validates row length
			*Returns:
				list of lists
		'''
		data = []
		numcols = len(text[0].split(','))

		for row in text:
			row = row.rstrip('\n')
			line = row.split(',')

			if len(line) != numcols:
				print("error here, not all rows have same number of columns. die.")
				sys.exit(1)

			data.append(line)

		self.numcols = numcols
		return data



	def train(self, train, centroids):

		self.centroids = centroids

		clean = self.clean(train)
		df = pd.DataFrame(data=clean)
		df.columns = [*df.columns[:-1], 'Class']

		self.train_data = clean
		self.train_df = df


		print(self.train_df)










































