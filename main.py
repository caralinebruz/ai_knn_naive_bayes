#!/usr/bin/env python3
import os
import sys
import getopt
import argparse

import KNN
from KNN import KNN

import NaiveBayes
from NaiveBayes import NaiveBayes

import KMeans
from KMeans import KMeans


def main(train, test, K, C, verbose, distance_function, centroids):
	print("OK. starting.")

	# if there are centroids, we are doing kmeans
	if not centroids:

		# evaluate the classifier method
		if K > 0:
			# value must be given for KNN
			# use knn method
			print("use knn classifier") 
			print("using distance function %s" % distance_function)

			if not test:
				print("did not submit a test file")

			k = KNN(verbose, K, distance_function)

			k.train(train)
			k.test(test)


		else:
			# use naive bayes 
			print("use naive bayes classifier")

			if not C:
				print("not using laplacian correction")
			else:
				print("using laplacian correction value %s" % C)

			if not test:
				print("did not submit a test file")

			n = NaiveBayes(verbose, C)
			n.train(train)

	else:
		print("use k-means")
		k = KMeans(verbose, distance_function)
		k.train(centroids, train)






# https://stackoverflow.com/questions/33499173/how-to-create-argument-of-type-list-of-pairs-with-argparse
def pair(arg):
	# For simplity, assume arg is a pair of integers
	# separated by a comma. If you want to do more
	# validation, raise argparse.ArgumentError if you
	# encounter a problem.
	return [int(x) for x in arg.split(',')]





if __name__ == '__main__':
	# USAGE: 

	# KNN 
	#   ./main.py -train data/inputs/knn.1.train.txt -test data/inputs/knn.1.test.txt -K 3

	# NAIVE BAYES
	#	./main.py -train data/inputs/nb.1.train.csv -test data/inputs/nb.1.test.csv
	# 	./main.py -train data/inputs/nb.1.train.csv -test data/inputs/nb.1.test.csv -C 2

	# KMEANS 
	#   ./main.py -train data/inputs/kmeans.1.txt -centroids 0,500 200,200 1000,1000
	# 	./main.py -train data/inputs/kmeans.2.txt -d manh -centroids 0,500,0 200,200,500 1000,1000,100


	# his 
	#	./main.py -train data/inputs/kmeans.2.txt -centroids 0,0,0 200,200,200 500,500,500



	# -train
	# -test
	# -K			: 
	# -C			: optional, default of 0 means don't use
	# -d			: distance function
	# -v			: optional
	# [centroids] 	: optional


	#
	# PARSE COMMAND LINE 
	#
	train_file = "" # file
	test_file = "" 	# file
	train_lines = []
	test_lines = []
	K = 0 			# num clusters
	C = 0 			# laplacian correction value (0 means unused)
	v_verbose = False

	# kmeans 
	centroids = []	# only used in kmeans
	distance_function = "e2" # in manhattan or euclidean squared (weighted)

	parser = argparse.ArgumentParser(description='KNN and or Naive Bayes parser')
	parser.add_argument('-train', type=argparse.FileType('r'), help="training file")
	parser.add_argument('-test', type=argparse.FileType('r'), help="test file")
	parser.add_argument('-K', help="knn num clusters")
	parser.add_argument('-C', help="optional naive bayes laplacian correction value")
	parser.add_argument('-v', action='store_true', help="minimize values as costs")

	# extra credit k-means
	parser.add_argument('-d', help="<manh,e2>")
	# -centroids  0,1 1,1 2,3
	parser.add_argument('-centroids', type=pair, nargs='+')
	

	args = parser.parse_args()
	print(args)

	if not args.train:
		print("no train file")
		sys.exit(1)
	else:
		train_file = args.train
		train_lines = args.train.readlines()

	if not args.test:
		print("no test file, assumes we will be doing pure kmeans")
	else:
		test_file = args.test
		test_lines = args.test.readlines()

	if args.K:
		K = int(args.K)
	if args.C:
		C = int(args.C)
	if args.d:
		distance_function = args.d
	if args.centroids:
		centroids = args.centroids
		print(centroids)
	if args.v:
		v_verbose = True

	# validation
	# illegal for c and k both to be > 0
	if K > 0 and C > 0:
		print("illegal arguments entered for K and C")
		sys.exit(1)

	main(train_lines, test_lines, K, C, v_verbose, distance_function, centroids)
