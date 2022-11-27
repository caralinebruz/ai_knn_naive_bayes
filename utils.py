# utils for things

# k-means and knn will use euclidean
# k-means will use manhattan as well

def euclidean_distance_sq(vector_1, vector_2):
	# define the euclidean distance (squared)
	# https://machinelearningmastery.com/distance-measures-for-machine-learning/
	euclidean_distance = sum((e1-e2)**2 for e1, e2 in zip(vector_1,vector_2))

	# print("euclidean_distance: %s" % euclidean_distance)
	return euclidean_distance


def average(seq):

	if len(seq) == 0:
		print("PROBLEM\n\n")
		print(seq)

	return float(sum(seq) / len(seq))