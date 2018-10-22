import numpy as np
import math
import matplotlib.pyplot as plt
from data_load import preprocessData, create_train_test_split
from knn import calculate_performance
from radius_nn import Sweep_Param


def main():

	x, y = preprocessData()  # Load data
	# K range to test
	Kmin = 3
	Kmax = 21

	iterations = 10
	# Create array of Ks to test
	K_arr = np.linspace(Kmin, Kmax, iterations)
	print(K_arr)
	Sweep_Param(x, y, K_arr, 0)  # Sweep through each k value and compute performance metrics

	# Pick best one and redo to plot decision boundary


if __name__ == "__main__":
	main()
