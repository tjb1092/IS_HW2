import numpy as np
import random
import math
import matplotlib.pyplot as plt


def data_load(fn, class_label, data_array):
	# Open text file and create and read in each line
	with open(fn) as f:
		contents = f.readlines()
	# for each line
	for data in contents:
		parsedData = data.split("\t")  # Split via tab deliminator
		parsedData[1] = parsedData[1][:-1]  #remove new-line character
		# Create data tuple by casting text to floats and add label
		data_array.append([float(parsedData[0]), float(parsedData[1]), class_label])
	return data_array


def create_train_test_split(x, y, percent):
	# Create Train/Test Split
	d_len = len(x)
	# Sample 80% of the indices
	train_index = random.sample(range(d_len - 1), math.ceil(percent*d_len))
	# Compute the remaining 20% of samples for the test set
	test_index = list(set(range(d_len-1)) - set(train_index))

	# Index out the train set from the total set
	x_train = x[train_index]
	y_train = y[train_index]
	# Index the test set
	x_test = x[test_index]
	y_test = y[test_index]

	# create data dict to pass up to other functions
	return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

def preprocessData():
		# Read in data
		fn_stressed = "stressed.txt"
		fn_ustressed = "unstressed.txt"

		data_array = []
		# Pass data array into functions to append everything together
		data_array = data_load(fn_stressed, 1, data_array)
		data_array = data_load(fn_ustressed, 0, data_array)

		d_array = np.asarray(data_array)  # Convert to numpy array

		# Split into X and y arrays
		x = d_array[:, 0:2]
		y = d_array[:,2]

		# Normalize W and H
		# Get max and min values
		maxs = np.max(x, axis=0)
		mins = np.min(x, axis=0)

		# Normalize x's between 0 and 1
		norm_x = (x - mins)/ (maxs-mins)

		return norm_x, y

def main():
	preprocessData()

if __name__ == "__main__":
	main()
