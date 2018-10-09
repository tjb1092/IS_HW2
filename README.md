# IS_HW2

Todo:


1.) Read in data. Either restructure the data itself or get fancy with text
parsing. (done)

1.5) Get ranges of all data and normalize everything to 0-1. (done)

2.) Assign class labels to data tuples. i.e. (W,H,S) (done)

2.5) Create a randomize function to shuffle the data around (done)
3.) Split into train/test 80/20 (done)

_______________________________

4.) KNN. In test set, take data point.

4.1) loop through each point in the training sample. Compute the euclidean distance.
4.2) Keep a list of the smallest k distances.
4.3) assign class to new sample, check if it is correct -> increment FP/TP, FN/TN rates.
