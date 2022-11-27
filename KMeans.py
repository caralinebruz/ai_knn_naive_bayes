''' K-Means Clustering Unsupervised Learning
'''
import sys
import numpy as np
import pandas as pd

import utils
from utils import euclidean_distance_sq, manhattan_distance



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


	def name_centroids(self):
		# name each centroid
		for i in range(len(self.centroids)):

			centroid_name = "C" + str(i)
			self.cluster_names[centroid_name] = self.centroids[i]

		if self.verbose:
			for k,v in self.cluster_names.items():
				print(k,v)


	def initialize_clusters(self):
		# # initialize empty list for the clusters, originally
		clusters = {}
		for name in self.cluster_names:
			clusters[name] = []

		self.clusters = clusters


	def get_distance_to_centroids(self, datapoint):
		distances = {}
		datapoint_name = datapoint[-1]
		data_vec = datapoint[:-1]
		data_vec = list(map(int, data_vec))

		for centroid, centroid_vec in self.cluster_names.items():

			centroid_vec = list(map(int, centroid_vec))

			if self.distance_function == "manh":
				distances[centroid] = manhattan_distance(data_vec, centroid_vec)
			else:
				distances[centroid] = euclidean_distance_sq(data_vec, centroid_vec)

		# evaluate the distances to all centroids
		smallest = float('inf')
		nearest_cluster = None

		for centroid_name, distance in distances.items():
			# print(centroid_name,distance)

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

			if not data:
				if self.verbose:
					print("warning, cluster is empty")

				# if there arent any items in the cluster
				# retain old centroid value
				new_centroids[cluster] = self.cluster_names[cluster]
				continue

			num_datapoints_in_cluster = len(data)

			# iterate over each datapoint in the cluster
			for item in data:

				# add each dimension of the datapoint
				for i in range(len(self.datapoints[item])):
					new_centroid[i] += int(self.datapoints[item][i])

			# calcualte average in each dimension
			new_centroid_avg = []
			for dimension in new_centroid:
				new_centroid_avg.append(dimension/num_datapoints_in_cluster)

			# print(new_centroid_avg)
			new_centroids[cluster] = new_centroid_avg

		if self.verbose:
			print("new centroid locations")
			for k,v in new_centroids.items():
				print(k,v)

		return new_centroids


	def save_dict_of_all_datapoints(self):
		datapoints = {}
		for item in self.train_data:
			name = item[-1]
			data = item[:-1]
			datapoints[name] = data

		self.datapoints = datapoints


	def iterate_to_convergence(self):
		''' contains stopping criteria for convergence of centroid locations
		'''
		y=0
		converge = False

		while not converge:
			if self.verbose:
				print("Iteration %s" % y)

			# reset clusters to empty
			self.initialize_clusters()

			# assign datapoints to clusters
			for i in range(len(self.train_data)):

				if y==500:
					print("break after 100 iterations")
					break

				datapoint = self.train_data[i]
				self.get_distance_to_centroids(datapoint)

			# update centroid locations
			new_centroids = self.update_centroids()

			# evaluate if we should stop
			change = False
			for k,v in new_centroids.items():
				for ok,ov in self.cluster_names.items():

					if k == ok:
						# my tolerance is simply to the nearest int
						int_v = list(map(int, v))
						int_ov = list(map(int, ov))

						if int_v != int_ov:
							change = True
						
			if not change:
				converge = True

				if self.verbose:
					print("reached stopping criteria after %s iterations." % y)
				
			else:
				# print("not converged, need to continue until converge")
				self.cluster_names = new_centroids

			y+=1

		return True


	def print_results(self):
		for cluster_name, items in self.clusters.items():
			item_set = set(items)
			print("%s = %s" % (cluster_name, items))

		for centroid, location in self.cluster_names.items():
			print("(%s)" % location)


	def train(self, centroids, train):

		# update centroids and num clusters to use
		self.centroids = centroids
		self.num_clusters = len(centroids)
		self.dimensions = len(centroids[0])

		self.name_centroids()
		
		clean = self.clean(train)


		if self.dimensions != self.numcols - 1:
			print("error, centroids are not same N-dimension as training data. abort")
			sys.exit(1)

		self.train_data = clean

		# also add a dictionary of all datapoints
		self.save_dict_of_all_datapoints()

		# do k-means until converge
		finished = self.iterate_to_convergence()
		if finished:
			self.print_results()








































