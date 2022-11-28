''' Naive Bayes Classifier
'''

# for average per row
import numpy as np
import pandas as pd

from pprint import pprint

class NaiveBayes:

	def __init__(self, verbose, correction):
		self.verbose = verbose
		self.correction = correction

		self.numcols = 0
		self.train_data = None
		self.train_df = None

		self.labels = []

		self.test_data = None
		self.test_df = None

		self.total_rows = 0
		self.total_rows_by_class = {}


		self.priors = {}


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
			print("label:%s rows:%s" % (label, num_rows_of_class))

			self.priors[label] = (num_rows_of_class / self.total_rows)
			self.total_rows_by_class[label] = num_rows_of_class

		for k,v in self.priors.items():
			print(k,v)


	def get_total_rows(self):
		''' *Returns:
				a single integer 
		'''
		return len(self.train_df.index)

	# def make_total_rows_by_class(self):
	# 	''' *Returns:
	# 			A dictionary of total rows by class
	# 	'''
	# 	for label in self.labels:

	# 		# get the number of rows with each class
	# 		num_rows_of_class = len(self.train_df[self.train_df.Class == label])
	# 		print("label:%s rows:%s" % (label, num_rows_of_class))

	# 		self.totals_by_class[str(label)] = num_rows_of_class


	def print_likelihoods(self):
		# if self.verbose:
		if True:
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
		print("make likelihoods")
		likelihoods = {}

		for label in self.labels:

			# select all rows for each class condition
			byclass_df = self.train_df.loc[self.train_df['Class'] == label]

			print(byclass_df)

			likelihoods[label] = {}

			# next, fill in predictor names
			for column in byclass_df.columns:

				
				if column != 'Class':

					#print("\tcolumn: %s " % column)

					likelihoods[label][str(column)] = {}

					# get unique values in that column
					unique_values = byclass_df[column].unique()
					# print("\t",unique_values)

					# for each unique value, save the frequency of the column
					for unique_value in unique_values:

						#print("\tfrequency of value %s in column %s" % (unique_value, column))

						# https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values
						freq = len(byclass_df.loc[byclass_df[column] == unique_value])

						# assign frequency 
						likelihoods[label][str(column)][str(unique_value)] = freq


		# once finished parsing the table, save
		self.likelihoods = likelihoods
		self.print_likelihoods()


	def train(self, data):
		''' Simply cleans the training dataset from the input file
		'''
		clean = self.clean(data)
		df = pd.DataFrame(data=clean)
		df.columns = [*df.columns[:-1], 'Class']

		self.train_data = clean
		self.train_df = df

		print(self.train_df)

		# get the total (for normalization)
		self.total_rows = self.get_total_rows()
		# self.make_total_rows_by_class()

		# print("total rows by class:")
		# for k,v in self.total_rows_by_class.items():
		# 	print(k,v)

		# get the priors
		self.make_prior_probabilities()

		# get the conditional probabilities frequencies
		self.make_likelihoods()


		''' likelihoods
			for each class, make a dictionary for their conditional probabilities
				each dictionary contains keys as predictor variable names
										 values as predictor variable values (dict)
										 		values of this dict will be frequency counts
		'''

	def classify(self, item):
		''' Attempt to classify the item 
		'''
		items_class = item[-1]
		items_data = item[:-1]

		data = items_data.copy()

		# data_vec = list(map(int, data_vec))

		# create the equation h(x) for each potential class (my case, T/F)

		for label in self.labels:


			h_x = self.priors[label]

			print("h_x prior = %s" % h_x)


			# each predictor has form:
			#
			# (num rows containing val, cond on class) / (total_by_class)
			#


			for i in range(len(data)):

				# value = data.pop()
				value = data[i]

				num_rows_containing_val_c_class = 0

				# test = self.likelihoods[label][str(i)][str(value)]

				# age = ages.get('Jim',0)
				test = self.likelihoods[label][str(i)].get(str(value))

				print("TEST VAL %s" % test)



				if value in self.likelihoods[label][str(i)].keys():
					print("in there")
																	 # class  # col  # unique val
					num_rows_containing_val_c_class = self.likelihoods[label][str(i)][str(value)]
				
				else:
					print("missing")

				numerator = num_rows_containing_val_c_class
				denominator = self.total_rows_by_class[label]

				print("\tmult %s / %s" % (numerator, denominator))
				h_x = h_x * (numerator / denominator)



			print("\th_x:%s" % h_x)

			# then, divide by total rows
			print("dividing by total rows: %s" % self.total_rows)
			h_x = h_x / self.total_rows

			print("\th_x:%s" % h_x)




	def test(self, data):
		''' Primary logic
		'''
		clean = self.clean(data)
		df = pd.DataFrame(data=clean)
		df.columns = [*df.columns[:-1], 'Class']

		self.test_df = df
		self.test_data = clean



		# # # get the unique values "actuals" in test set
		# self.prepare_labels()

		print("starting test")
		for row in self.test_data[:1]:

			print(row)

			if len(row) != self.numcols:
				print("test row has different N-dimensions from train set. abort")
				sys.exit(1)

			self.classify(row)













































