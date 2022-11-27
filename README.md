# ai_knn_naive_bayes
k-nearest-neighbors classifier with option for naive bayes classifier


### Instructions

This package has required dependencies. I built this using `python-3.9.7`

1. create a virtual environment
2. `pip3 install -r requirements.txt`


## Extra Credit K-Means

Default value for distance metric is euclidean squared distance. Optionally, you may specify which metric to use with his suggested `-d` flag.

Command line for K-means requires flag `-centroids`:

`./main.py -train data/inputs/kmeans.1.txt -centroids 0,500 200,200 1000,1000`

### Graders
I matched all his expected outputs for euclidean squared distance.

I match his expected outputs for manhattan distance, except I'm not sure which centroids he used for `kmeans.1.` manhattan distance. I match correctly for the centroids he listed using manhattan distance in `kmeans.2`.
