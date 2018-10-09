import numpy as np
from data_load import preprocessData
import math
import matplotlib.pyplot as plt


def knn(data, k):
	print(type(data["x_test"]))


	# for each test data point,
	ybar = np.zeros(data["y_test"].shape)
	for i, x_test in enumerate(data["x_test"]):
		#loop through each training point.
		dist_list = []
		max_dist = 0.0
		for j, x_train in enumerate(data["x_train"]):
			# compute euclidean distance between points
			dist = math.sqrt((x_test[0]-x_train[0])**2 + (x_test[1] - x_train[1])**2)
			if len(dist_list) == k:
				if dist < max_dist:
					dist_list.pop()
					dist_list.append([dist, data["y_train[j]"]])
					dist_list.sort()
			else:
				dist_list.append([dist, data["y_train"][j]])
				dist_list.sort()

		one_cnt = 0
		zero_cnt = 0
		for element in dist_list:
			if element[1] == 0:
				zero_cnt += 1
			else:
				one_cnt += 1

		if one_cnt > zero_cnt:
			ybar[i] = 1  # Assume 0 otherwise

	print(ybar == data["y_test"])

def main():
	data = preprocessData()
	plt.scatter(data["x_train"][:,0], data["x_train"][:,1], c=data["y_train"])
	plt.show()

	knn(data, 3)


if __name__ == "__main__":
	main()
