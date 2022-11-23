#!/usr/bin/env python3
import os
import sys
import getopt
import argparse

import KNN
from KNN import KNN

import NaiveBayes
from NaiveBayes import NaiveBayes


def main(train, test, K, C, verbose):
	print("OK. starting.")

	# evaluate the classifier method
	if K > 0:
		# value must be given for KNN
		# use knn method
		print("use knn classifier") 

		if not test:
			print("did not submit a test file")

		k = KNN(verbose, K)

	else:
		# use naive bayes 
		print("use naive bayes classifier")

		if not test:
			print("did not submit a test file")

		n = NaiveBayes(verbose, C)
		n.train(train)



	# need to figure out what will be the input for kmeans??






if __name__ == '__main__':
	# USAGE: 
	# 	./main.py -df .9 -tol 0.0001 data/input/maze.txt

	# knn 
	#   learn -train train.txt -test test.txt -K 3

	# naive bayes with no laplacian, but with kmeans args ??
	#	learn -train train.txt -test test.txt [[x0,y0], [x1,y2], ...]

	# kmeans 
	#   learn -train some-input.txt 0,500 200,200 1000,1000

	# -train
	# -test
	# -K			: 
	# -C			: optional, default of 0 means don't use
	# -v			: optional
	# [centroids] 	: optional



	#
	# PARSE COMMAND LINE 
	#
	train_file = "" # file
	test_file = "" 	# file
	K = 0 			# num clusters
	C = 0 			# laplacian correction value (0 means unused)
	v_verbose = False
	centroids = []	# only used in kmeans

	train_lines = []
	test_lines = []

	parser = argparse.ArgumentParser(description='KNN and or Naive Bayes parser')

	parser.add_argument('-train', type=argparse.FileType('r'), help="training file")
	parser.add_argument('-test', type=argparse.FileType('r'), help="test file")
	parser.add_argument('-K', help="knn num clusters")
	parser.add_argument('-C', help="optional naive bayes laplacian correction value")
	parser.add_argument('-v', action='store_true', help="minimize values as costs")


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
		K = args.K
	if args.C:
		C = args.C

	if args.v:
		v_verbose = True

	# validation
	# illegal for c and k both to be > 0
	if K > 0 and C > 0:
		print("illegal arguments entered for K and C")
		sys.exit(1)


	main(train_lines, test_lines, K, C, v_verbose)
