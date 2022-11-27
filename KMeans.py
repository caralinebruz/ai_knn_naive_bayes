''' K-Means Clustering Unsupervised Learning
'''
import sys
import numpy as np
import pandas as pd

import utils
from utils import euclidean_distance_sq, average




class KMeans:

	def __init__(self, verbose, distance_function):
		self.verbose = verbose
		self.distance_function = distance_function
		self.centroids = []
		self.num_clusters = 1
		self.train_df = None
		self.train_data = None
		self.numcols = 0
		self.dimensions = 0

		self.clusters = {}
		self.cluster_names = {}
		self.datapoints = {}


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


	def get_distance_to_centroids(self, datapoint):
		print(datapoint)

		distances = {}
		datapoint_name = datapoint[-1]
		data_vec = datapoint[:-1]
		data_vec = list(map(int, data_vec))

		for centroid, centroid_vec in self.cluster_names.items():

			centroid_vec = list(map(int, centroid_vec))
			distances[centroid] = euclidean_distance_sq(data_vec, centroid_vec)

		# evaluate the distances to all centroids
		smallest = float('inf')
		nearest_cluster = None

		for centroid_name, distance in distances.items():
			#print(centroid_name,distance)

			if distance < smallest:
				smallest = distance 
				nearest_cluster = centroid_name

		# assign to the nearest cluster
		self.clusters[nearest_cluster].append(datapoint_name)


	def update_centroids(self):
		# Recompute the centroids as an average of all the points assigned


		new_centroids = {}
		for cluster, data in self.clusters.items():

			# initialize a new centroid [0, 0, 0, 0]
			new_centroid = [0] * self.dimensions

			print("starting cluster %s" % cluster)

			if not data:
				print("warning, cluster is empty")

			num_datapoints_in_cluster = len(data)

			# iterate over each datapoint in the cluster
			for item in data:
				#print(item)
				print(self.datapoints[item])
				#clusters_data.append(self.datapoints[item])

				# add each dimension of the datapoint
				for i in range(len(self.datapoints[item])):
					new_centroid[i] += int(self.datapoints[item][i])

			# cluster_df = pd.DataFrame(clusters_data)
			# print(cluster_df)

			print("sum")
			print(new_centroid)

			print("avg")
			new_centroid_avg = []
			for dimension in new_centroid:
				new_centroid_avg.append(dimension/num_datapoints_in_cluster)

			print(new_centroid_avg)


			# print(cluster_df.mean(axis=0))




	def save_dict_of_all_datapoints(self):
		datapoints = {}
		for item in self.train_data:
			name = item[-1]
			data = item[:-1]

			datapoints[name] = data

		self.datapoints = datapoints

		for k,v in self.datapoints.items():
			print(k,v)



	def train(self, centroids, train):

		# update centroids and num clusters to use
		self.centroids = centroids
		self.num_clusters = len(centroids)
		self.dimensions = len(centroids[0])
		self.initialize_clusters()

		clean = self.clean(train)
		df = pd.DataFrame(data=clean)
		df.columns = [*df.columns[:-1], 'Name']




		if self.num_clusters != self.numcols - 1:
			print("error, centroids are not same N-dimension as training data. abort")
			sys.exit(1)

		self.train_data = clean
		self.train_df = df
		# also add a dictionary of all datapoints
		self.save_dict_of_all_datapoints()


		print(self.train_df)


		# assign clusters
		# for each of the datapoints, measure the distance to each of the centroids
		# pick the relevant centroid index:
		#		euclidean distance -> smallest number
		#		manhattan distance -> smallest number

		for i in range(len(self.train_data)):

			datapoint = self.train_data[i]


			if i <= 10:
			#if True:
	
				self.get_distance_to_centroids(datapoint)


		# evaluate the clusters
		for k,v in self.clusters.items():
			print(k,v)


		self.update_centroids()









































