''' K-Nearest Neighbors Classifier
'''
import sys
# for average per row
import numpy as np
import pandas as pd


def dot_prod(vector_1, vector_2):
	return float(sum(float(x) * float(y) for x, y in zip(vector_1, vector_2)))

def mag(vec):
	return float(sqrt(dot_prod(vec, vec)))


def euclidean_distance_sq(vector_1, vector_2):
	# define the euclidean distance (squared)
	# https://machinelearningmastery.com/distance-measures-for-machine-learning/
	euclidean_distance = sum((e1-e2)**2 for e1, e2 in zip(vector_1,vector_2))

	print("euclidean_distance: %s" % euclidean_distance)
	return euclidean_distance




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

		clean = self.clean(data)
		df = pd.DataFrame(data=clean)
		# df = df.replace('\n','', regex=True)
		df.columns = [*df.columns[:-1], 'Class']

		print(df)
		self.train_data = clean
		self.train_df = df


	def get_distances(self, test_row):

		# for each row in the training data
		# get the distances
		print("using test data set row")
		print(test_row)
		test_set = test_row[:-1]
		test_set = list(map(int, test_set))


		print("measuring nearest rows...")
		distances = []
		for x in range(len(self.train_data)):

			train_set = self.train_data[x][:-1]
			train_set = list(map(int, train_set))


			print(train_set)
			print(test_set)

			if len(test_set) != len(train_set):
				print("inputs do not have same number of cols. abort")
				sys.exit(1)

			distances.append(euclidean_distance_sq(test_set, train_set))



		return distances


	def vote(self, distances_df):

		# select the K smallest distances (min neighbors)
		closest_df = distances_df.nsmallest(self.K, 'distances')
		print(closest_df)

		# determine the class based on weighted 1/distance

		# https://stackoverflow.com/questions/33768122/python-pandas-dataframe-how-to-multiply-entire-column-with-a-scalar
		closest_df['weighted_distance'] = closest_df['distances'].apply(lambda x: 1/x)
		print(closest_df)

		# sum weighted distances by group 'Class'
		# https://stackoverflow.com/questions/39922986/how-do-i-pandas-group-by-to-get-sum
		results_df = closest_df.groupby(['Class'])['weighted_distance'].sum().reset_index()
		#results_df = closest_df.groupby(['Class'])['weighted_distance'].sum()

		# then, select the class with the largest vote
		# selected = 
		# 

		# rename result column

		print("results")
		print(results_df)

		# print(results_df[results_df.weighted_distance == results_df.weighted_distance.max()])
		# top = results_df[results_df.weighted_distance == results_df.weighted_distance.max()]
		# print(top['Class'])

		selected_class = results_df.iloc[0]['Class']
		print("selected class for row: %s" % selected_class)

		return selected_class



	def test(self, data):

		clean = self.clean(data)
		df = pd.DataFrame(data=clean)
		# df = df.replace('\n','', regex=True)
		df.columns = [*df.columns[:-1], 'Class']

		print(df)
		self.test_df = df
		self.test_data = clean


		print("testing....")

		# for index, row in df.iterrows():

		# 	if index == 1:
		# 		print(row)


		i = 0
		# iterate over the test data
		for i in range(len(clean)):

			# get the distances from this test point to all training set data points

			# make a copy of the train dataframe
			train_df_copy = self.train_df.copy()

			if i == 0:

				d = self.get_distances(clean[i])

				print(d)

				train_df_copy['distances'] = d


				# represents the distances from the test set
				# to every point in the training set
				print(train_df_copy)

				# step 2, voting, weighted
				self.vote(train_df_copy)




















































