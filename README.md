# ai_knn_naive_bayes
k-nearest-neighbors classifier with option for naive bayes classifier, includes k-means computation


## Extra Credit K-Means

Default distance metric is euclidean squared. Optionally, you may specify which metric to use with his suggested `-d` flag.

:orange_circle: REQUIRED: Command line requires flag `-centroids`:

`./main.py -train data/inputs/kmeans.1.txt -centroids 0,500 200,200 1000,1000`


## Non-Cims Instructions

This package has required dependencies. I built this using `python-3.9.7`

1. create a virtual environment
2. `pip3 install -r requirements.txt`


## CIMS Instructions

This code has been tested on `access.cims.nyu.edu`. Running this code on CIMS machines only requires `numpy`. Do not pip install requirements.txt, CIMS has not been updated and will not find a matching distribution of the version of pandas and numpy listed in the requirements file. Instead, simply run:

`pip3 install numpy`


### Executable
`./main.py -train data/inputs/knn.1.train.txt -test data/inputs/knn.1.test.txt -K 3`

will output the following:
```
use knn
using distance function e2
want=A got=A
want=B got=A
want=A got=A
want=A got=B
want=B got=B
want=B got=B
Label=A Precision=2/3 Recall=2/3
Label=B Precision=2/3 Recall=2/3
```




### Note for Graders
I matched all his expected outputs for euclidean squared distance.

I match his expected outputs for manhattan distance, except I'm not sure which centroids he used for `kmeans.1.` manhattan distance. I match correctly for the centroids he listed using manhattan distance in `kmeans.2`.


### References
I have written all of this code. References, when used, are noted in comments of method which they are used in.
