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

4.1) loop through each point in the training sample. Compute the euclidean distance. (done)
4.2) Keep a list of the smallest k distances. (done)
4.3) assign class to new sample, check if it is correct (done)

Things I need to calculate:
- Hit Rate, Sensitivity, Specificity, PPV, NPV (done)

4.5) plot points w/ heuristically defined boundary line.
4.5.1) scatter training points with dots. color by class
4.5.2) scatter test points as slightly bigger x's. color by predicted class
4.5.3) loop through a grid of points in the range of H/W. Use KNN to pick class.
4.5.4) scatter grid as small dots. color by predicted class.   

5.) Perceptron

5.1) Initialize weight matrix (two values for 2D space) so line is intersecting
the data cloud somewhere (take two points from each class and use point slope) (done)
5.2) for each epoch, loop through each training point and calculate that (done)
5.2.5) adjust weights via formulas (done. Check that dot product is working right)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55555
10/18

Todo:
- Validate perceptron operation.
- Add Error Rate graph (once per 50 epochs) over training and testing data. This should be able to be done in a single line as an array operator.s
- Plot H-W plane for perceptron
- Add legend to the KNN HW plane
- radius -NN algo
- Augment the KNN for relative distance weighting
