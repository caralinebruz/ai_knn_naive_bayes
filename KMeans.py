''' K-Means Clustering Unsupervised Learning
'''
import sys
import numpy as np
import pandas as pd

import utils
from utils import euclidean_distance_sq




class KMeans:

	def __init__(self, verbose, distance_function):
		self.verbose = verbose
		self.distance_function = distance_function
		self.centroids = []
		self.num_clusters = 1
		self.train_df = None
		self.train_data = None
		self.numcols = 0

		self.clusters = {}
		self.cluster_names = {}


	def clean(self, text):
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


	# def initialize_clusters(self):

	# 	for 

	def get_distances(self, test_row):

		test_set = test_row[:-1]
		test_set = list(map(int, test_set))


		# print("measuring nearest rows...")
		distances = []
		for x in range(len(self.train_data)):

			train_set = self.train_data[x][:-1]
			train_set = list(map(int, train_set))


			# print(train_set)
			# print(test_set)

			if len(test_set) != len(train_set):
				print("inputs do not have same number of cols. abort")
				sys.exit(1)

			distances.append(euclidean_distance_sq(test_set, train_set))

		return distances


	def get_distance_to_centroids(self, datapoint):
		print(datapoint)

		distances = {}
		datapoint_name = datapoint[-1]
		data_vec = datapoint[:-1]
		data_vec = list(map(int, data_vec))

		for centroid, centroid_vec in self.cluster_names.items():
			print(centroid, centroid_vec)

			centroid_vec = list(map(int, centroid_vec))

			distances[centroid] = None

			# d = euclidean_distance_sq(data_vec, centroid_vec)

			distances[centroid] = euclidean_distance_sq(data_vec, centroid_vec)

			#print(distances[centroid])


		# evaluate the distances to all centroids
		smallest = float('inf')
		nearest_cluster = None

		for centroid_name, distance in distances.items():
			print(centroid_name,distance)

			print("%s < %s ?" % (distance, smallest))
			if distance < smallest:
				smallest = distance 
				nearest_cluster = centroid_name


		print("choose cluster %s for %s" % (nearest_cluster, smallest))


		print(self.clusters[nearest_cluster])

		# assign to the nearest cluster
		self.clusters[nearest_cluster].append(datapoint_name)

		print(self.clusters[nearest_cluster])



	def initialize_clusters(self):
		# name each centroid
		for i in range(len(self.centroids)):

			centroid_name = "C" + str(i)
			print(centroid_name)
			self.cluster_names[centroid_name] = self.centroids[i]

		for k,v in self.cluster_names.items():
			print(k,v)

		# # initialize empty list for the clusters, originally
		clusters = {}
		for name in self.cluster_names:
			clusters[name] = []

		self.clusters = clusters


	def train(self, centroids, train):

		# update centroids and num clusters to use
		self.centroids = centroids
		self.num_clusters = len(centroids)
		self.initialize_clusters()

		clean = self.clean(train)
		df = pd.DataFrame(data=clean)
		df.columns = [*df.columns[:-1], 'Name']


		if self.num_clusters != self.numcols - 1:
			print("error, centroids are not same N-dimension as training data. abort")
			sys.exit(1)

		self.train_data = clean
		self.train_df = df


		print(self.train_df)


		# assign clusters
		# for each of the datapoints, measure the distance to each of the centroids
		# pick the relevant centroid index:
		#		euclidean distance -> smallest number
		#		manhattan distance -> smallest number

		for i in range(len(self.train_data)):

			datapoint = self.train_data[i]


			if i <= 10:
	
				self.get_distance_to_centroids(datapoint)


		# evaluate the clusters
		for k,v in self.clusters.items():
			print(k,v)









































