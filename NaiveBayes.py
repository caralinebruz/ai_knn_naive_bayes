''' Naive Bayes Classifier
'''
import sys

import numpy as np
import pandas as pd

from pprint import pprint

class NaiveBayes:

	def __init__(self, verbose, correction):
		self.verbose = verbose
		self.correction = correction
		self.domain = {}
		self.numcols = 0
		self.labels = []
		self.train_data = None
		self.train_df = None
		self.test_data = None
		self.test_df = None

		self.total_rows = 0
		self.total_rows_by_class = {}
		self.priors = {}
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
		unique_labels = list(map(str, unique_labels))

		self.labels = unique_labels


	def make_prior_probabilities(self):
		''' Uses the train dataframe to get priors
		'''
		labels = self.train_df.Class.unique()
		self.labels = labels

		for label in self.labels:

			# get the number of rows with each class
			num_rows_of_class = len(self.train_df[self.train_df.Class == label])
			# print("label:%s rows:%s" % (label, num_rows_of_class))

			self.priors[label] = (num_rows_of_class / self.total_rows)
			self.total_rows_by_class[label] = num_rows_of_class


	def get_total_rows(self):
		''' *Returns:
				a single integer 
		'''
		return len(self.train_df.index)


	def make_domain(self):
		''' Save the domain of each predictive attribute
		'''
		domain = {}
		# for each predictor column
		# get the count of unique values in the row.
		for x in range(self.numcols - 1):

			d = len(self.train_df[x].unique())
			domain[str(x)] = d

		self.domain = domain


	def print_likelihoods(self):
		# prints the frequency counts dictionary
		if self.verbose:
			for cl, v in self.likelihoods.items():
				print("{'%s':" % cl)
				for col, uniquevals in self.likelihoods[cl].items():
					print("\t{'%s':" % col)
					print("\t\t", uniquevals)
					print("\t}")
				print("}")


	def make_likelihoods(self):
		''' Creates the conditional probabilities

			*Returns: -> dict
				L1 = class names
					L2 = predictor names
						L3 = predictor values
							L4 = frequency counts

				T:  0 : { 1:2,  2:2 }
					1 : { 3:4 }
					2 : { 0:1, 1:2, 2:1}

				L1 = Class
				L2 = Column
				L3 = value in column
				L4 = count of value in column
		'''
		likelihoods = {}

		for label in self.labels:

			# select all rows for each class condition
			byclass_df = self.train_df.loc[self.train_df['Class'] == label]

			# print(byclass_df)
			likelihoods[label] = {}

			# next, fill in predictor names
			for column in byclass_df.columns:
				if column != 'Class':

					likelihoods[label][str(column)] = {}
					unique_values = byclass_df[column].unique()

					# for each unique value, save the frequency of the column
					for unique_value in unique_values:

						# https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values
						freq = len(byclass_df.loc[byclass_df[column] == unique_value])

						# assign frequency 
						likelihoods[label][str(column)][str(unique_value)] = freq


		# once finished parsing the table, save
		self.likelihoods = likelihoods
		# self.print_likelihoods()


	def train(self, data):
		''' Simply cleans the training dataset from the input file
		'''
		clean = self.clean(data)
		df = pd.DataFrame(data=clean)
		df.columns = [*df.columns[:-1], 'Class']

		self.train_data = clean
		self.train_df = df

		# get the total (for normalization)
		self.total_rows = self.get_total_rows()

		# get the priors
		self.make_prior_probabilities()

		# get the conditional probabilities frequencies
		self.make_likelihoods()

		# also get domain of each predictor
		self.make_domain()


	def _argmax(self, h_x):
		''' Sifts through args calculated
			*Returns:
				prediction of class based on argmax
		'''
		largest = float('-inf')
		selected_class = None

		for label in h_x.keys():

			if h_x[label] > largest:
				largest = h_x[label]
				selected_class = label

		return selected_class


	def classify(self, item):
		''' Makes a prediction on the items class

			*Returns:
				A selected class 
		'''
		items_class = item[-1]
		items_data = item[:-1]
		data = items_data.copy()
		arg = {}

		for label in self.labels:

			h_x = self.priors[label]

			if self.verbose:
				print("P(C=%s) = [%s]" % (label, h_x))

			for i in range(len(data)):

				# extract the test value of the current predictor
				value = data[i]
				
				# get look up the frequency, if available
				test = self.likelihoods[label][str(i)].get(str(value))

				# if it is available, populate it
				if value in self.likelihoods[label][str(i)].keys():
																	 # class  # col  # unique val
					num_rows_containing_val_c_class = self.likelihoods[label][str(i)][str(value)]
				else:
					num_rows_containing_val_c_class = 0

				# add laplacian correction (if any) to numerator
				numerator = num_rows_containing_val_c_class + self.correction
				denominator = self.total_rows_by_class[label] + (self.correction * self.domain[str(i)])

				if self.verbose:
					print("P(A%s=%s | C=%s) %s / %s" % (i, value, label, numerator, denominator))

				h_x = h_x * (numerator / denominator)


			if self.verbose:
				print("NB(C=%s) = %s" % (label, h_x))

			# stash the computation for argmax
			arg[label] = h_x

		return self._argmax(arg)


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
		# if self.verbose:
		# 	print("actuals")
		# 	for k,v in self.actual.items():
		# 		print(k,v)
		# 	print("predicted")
		# 	for k,v in self.predicted.items():
		# 		print(k,v)
		# 	print("correct")
		# 	for k,v in self.correct_prediction.items():
		# 		print(k,v)

		for label in self.labels:
			print("Label=%s Precision=%s/%s Recall=%s/%s" % (label, self.correct_prediction[label], self.predicted[label], self.correct_prediction[label], self.actual[label]))


	def test(self, data):
		''' Calls classify, evaluates the predicted class

			finally, prints results of confusion matrix
		'''
		clean = self.clean(data)
		df = pd.DataFrame(data=clean)
		df.columns = [*df.columns[:-1], 'Class']

		self.initialize_precision_recall()

		self.test_df = df
		self.test_data = clean

		for row in self.test_data:

			# print(row)
			actual_class = row[-1]

			if len(row) != self.numcols:
				print("test row has different N-dimensions from train set. abort")
				sys.exit(1)

			# make a prediction
			prediction = self.classify(row)

			# evaluate that prediction
			if self.verbose:
				if prediction != actual_class:
					print("fail: got '%s' != want '%s'" % (prediction, actual_class))
				else:
					print("match: '%s'" % prediction)


			self.predicted[prediction] += 1
			self.actual[actual_class] += 1

			if actual_class == prediction:
				self.correct_prediction[actual_class] += 1


		self.print_precision_recall()









































