import numpy as np
import math
import matplotlib.pyplot as plt
from data_load import preprocessData

def perceptron_train(data, epochs, LR):

	# First, initialize weights. For 2D case, there are three weights.

	# Pick two points, one in class 1, one in class 0.
	p1 = data["x_train"][0]
	if data["y_train"][0] == 1:
		lookfor = 0
	else:
		lookfor = 1

	i = 1
	found = False
	while not found:
		if data["y_train"][i] == lookfor:
			p2 = data["x_train"][i]
			found = True
		i += 1

	# pick a line that bisects the midpoint of those two points with a slope
	# perpendicular to the
	wi0 = 1.0
	wi2 = (2.0 * (p2[0] - p1[0])) / ((p1[1] - p2[1]) * (p1[0] + p2[0]) - (p1[1] + p2[1]) * (p2[0] - p1[0]))
	wi1 = ((p2[1] - p1[1]) / (p2[0] - p1[0])) * wi2

	wi = np.array([wi1, wi2])
	

def main():
	data = preprocessData()
	perceptron_train(data, 3)


if __name__ == "__main__":
	main()
