''' K-Nearest Neighbors Classifier
'''
import sys

import numpy as np
import pandas as pd

import utils
from utils import euclidean_distance_sq



class KNN:

	def __init__(self, verbose, K, distance_function):
		self.verbose = verbose
		self.K = K
		self.distance_function = distance_function
		self.train_data = None
		self.train_df = None
		self.test_data = None
		self.train_df = None
		self.numcols = 0
		self.labels = []
		self.actual = {}
		self.predicted = {}
		self.correct_prediction = {}

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


	def train(self, data):
		''' Simply cleans the training dataset from the input file
		'''
		clean = self.clean(data)
		df = pd.DataFrame(data=clean)
		df.columns = [*df.columns[:-1], 'Class']

		self.train_data = clean
		self.train_df = df


	def get_distances(self, test_row):
		''' Get distance from current row, to all training set points
			*Returns:
				an array of distances
		'''
		test_set = test_row[:-1]
		test_set = list(map(int, test_set))

		# print("measuring nearest rows...")
		distances = []
		for x in range(len(self.train_data)):

			train_set = self.train_data[x][:-1]
			train_set = list(map(int, train_set))

			if len(test_set) != len(train_set):
				print("inputs do not have same number of cols. abort")
				sys.exit(1)

			distances.append(euclidean_distance_sq(test_set, train_set))

		return distances


	def vote(self, distances_df):
		''' Determines the class based on weighted 1/distance
			*Returns:
				Selected class of the datapoint
		'''
		if self.verbose:
			print("overall results:")
			print(distances_df)

		# select the K smallest distances (min neighbors)
		closest_df = distances_df.nsmallest(self.K, 'distances')

		# if point is exactly on a datapoint in the training set
		# immediately return that class of training set
		exact = closest_df.loc[closest_df['distances'] == 0]

		if not exact.empty:
			selected_class = exact.iloc[0]['Class']
			if self.verbose:
				print("found exact match, selected class for row: %s" % selected_class)
			return selected_class

		# normalize the distances
		# except if the distance is zero, avoid divbyzero, use 1
		closest_df['weighted_distance'] = closest_df['distances'].apply(lambda x: 1/x if x>0 else 1)

		if self.verbose:
			print("apply 1/distance:")
			print(closest_df)

		# sum weighted distances by group 'Class'
		# https://stackoverflow.com/questions/39922986/how-do-i-pandas-group-by-to-get-sum
		results_df = closest_df.groupby(['Class'])['weighted_distance'].sum().reset_index()
		results_df = results_df.sort_values('weighted_distance', ascending=False)
		
		if self.verbose:
			print("results")
			print(results_df)

		# take top most votes
		selected_class = results_df.iloc[0]['Class']
		if self.verbose:
			print("selected class for row: %s" % selected_class)

		return selected_class


	def prepare_labels(self):
		''' Combine labels seen between the test and
			train sets (for precision recall use)
		'''

		labels_in_train = self.train_df.Class.unique()
		labels_in_test = self.test_df.Class.unique()

		labels = []
		labels.extend(labels_in_test)
		labels.extend(labels_in_train)

		unique_labels = sorted(list(set(labels)))
		self.labels = unique_labels


	def initialize_precision_recall(self):
		''' Makes dictionaries for each label
			for tracking prediction scores
		'''
		actual = {}
		predicted = {}
		correct_prediction = {}
		for label in self.labels:
			actual[label] = 0
			predicted[label] = 0
			correct_prediction[label] = 0

		self.actual = actual
		self.predicted = predicted
		self.correct_prediction = correct_prediction


	def print_precision_recall(self):
		''' Prints performance of classifier
		'''
		if self.verbose:
			print("actuals")
			for k,v in self.actual.items():
				print(k,v)
			print("predicted")
			for k,v in self.predicted.items():
				print(k,v)
			print("correct")
			for k,v in self.correct_prediction.items():
				print(k,v)


		for label in self.labels:
			print("Label=%s Precision=%s/%s Recall=%s/%s" % (label, self.correct_prediction[label], self.predicted[label], self.correct_prediction[label], self.actual[label]))


	def test(self, data):
		''' Primary logic for the knn algorithm
		'''
		clean = self.clean(data)
		df = pd.DataFrame(data=clean)
		df.columns = [*df.columns[:-1], 'Class']

		self.test_df = df
		self.test_data = clean

		# # get the unique values "actuals" in test set
		self.prepare_labels()
		self.initialize_precision_recall()

		i = 0
		for i in range(len(clean)):

			# make a copy of the train dataframe
			train_df_copy = self.train_df.copy()

			# separate the data from the class
			item = clean[i]
			items_true_class = item[-1]

			if self.verbose:
				print("using test row:")
				print(item)

			# step 1, get the distances from this test point to all training set data points 
			d = self.get_distances(item)

			train_df_copy['distances'] = d

			# step 2, voting, weighted
			my_classification = self.vote(train_df_copy)

			print("want=%s got=%s" % (items_true_class, my_classification))

			self.predicted[my_classification] += 1
			self.actual[items_true_class] += 1

			if items_true_class == my_classification:
				self.correct_prediction[items_true_class] += 1


		self.print_precision_recall()
















































