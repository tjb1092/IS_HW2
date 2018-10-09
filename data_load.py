import numpy as np
import random
import math
import matplotlib.pyplot as plt

def data_load(fn, class_label, data_array):
	with open(fn) as f:
		stress = f.readlines()

	for data in stress:
		parsedData = data.split("\t")
		parsedData[1] = parsedData[1][:-1]
		data_array.append([float(parsedData[0]), float(parsedData[1]), class_label])
	return data_array


def preprocessData():
		# Read in data
		fn_stressed = "stressed.txt"
		fn_ustressed = "unstressed.txt"

		data_array = []

		data_array = data_load(fn_stressed, 1, data_array)
		data_array = data_load(fn_ustressed, 0, data_array)

		d_array = np.asarray(data_array)  # Convert to numpy array

		# Split into X and y arrays
		x = d_array[:, 0:2]
		y = d_array[:,2]

		d_len = len(x)
		# Normalize W and H

		# Get max and min values
		maxs = np.max(x, axis=0)
		mins = np.min(x, axis=0)

		# Normalize x's between 0 and 1
		norm_x = (x - mins)/ (maxs-mins)

		# Create Train/Test Split
		train_percent = 0.8
		train_index = random.sample(range(d_len - 1), math.ceil(train_percent*d_len))
		# Compute the remaining 20% of samples for the test set
		test_index = list(set(range(len(d_array-1))) - set(train_index))

		x_train = norm_x[train_index]
		y_train = y[train_index]

		x_test = norm_x[test_index]
		y_test = y[test_index]

		# create data dict to pass up to other functions
		data = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
		return data

def main():
	preprocessData()

if __name__ == "__main__":
	main()
