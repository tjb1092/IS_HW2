import numpy as np
import math
import matplotlib.pyplot as plt
from data_load import preprocessData, create_train_test_split

def calculate_performance(ybar, y):
	Q_00, Q_11, Q_10, Q_01 = 0., 0., 0., 0.

	for i, ybari in enumerate(ybar):
		if ybari == y[i]:
			# Correct predictions
			if ybari == 1:
				Q_11 += 1.
			else:
				Q_00 += 1.
		else:
			# Misclassifications
			if ybari == 1:
				Q_01 += 1.
			else:
				Q_10 += 1.

	Hit_Rate = (Q_11 + Q_00) / (Q_11 + Q_00 + Q_10 + Q_01)
	Sensitivity = Q_11 / (Q_11 + Q_10)
	Specificity = Q_00 / (Q_01 + Q_00)
	PPV = Q_11 / (Q_11 + Q_01)
	NPV = Q_00 / (Q_00 + Q_10)
	print(Hit_Rate, Sensitivity, Specificity, PPV, NPV)
	return Hit_Rate, Sensitivity, Specificity, PPV, NPV

def knn(data, k):

	# for each test data point,
	ybar = np.zeros(len(data["x_test"]))
	for i, x_test in enumerate(data["x_test"]):
		#loop through each training point.
		dist_list = []
		for j, x_train in enumerate(data["x_train"]):
			# compute euclidean distance between points
			dist = math.sqrt((x_test[0]-x_train[0])**2 + (x_test[1] - x_train[1])**2)

			if len(dist_list) == k:
				if dist < dist_list[-1][0]:
					dist_list.pop()
					dist_list.append([dist, data["y_train"][j]])
					dist_list.sort()
			else:
				dist_list.append([dist, data["y_train"][j]])
				dist_list.sort()

		one_cnt , zero_cnt= 0, 0
		for element in dist_list:
			#  Weight each neighbor by 1/dist.
			if element[1] == 0:
				zero_cnt += 1*(1/element[0])
			else:
				one_cnt += 1*(1/element[0])
		if one_cnt > zero_cnt:
			ybar[i] = 1  # Assume 0 otherwise

	return ybar


def main():
	k = 13
	x, y = preprocessData()
	data = create_train_test_split(x, y, 0.8)

	fig_scatter, ax_scatter = plt.subplots()  # Get data for H-W Plane

	ybar = knn(data, k)

	ax_scatter.scatter(data["x_train"][:,0], data["x_train"][:,1], c=data["y_train"], s=10, marker="D")
	ax_scatter.scatter(data["x_test"][:,0], data["x_test"][:,1], c=ybar, s=60, marker="x")


	d_testx = np.arange(0,1.01,1e-2)
	d_test_grid = np.array(np.meshgrid(d_testx, d_testx)).T.reshape(-1,2)  # Create grid of test-points
	data["x_test"] = d_test_grid  # Change over X's to test grid, don't care about y's

	ybar_grid = knn(data, k)

	ax_scatter.scatter(data["x_test"][:,0], data["x_test"][:,1], c=ybar_grid, s=1)
	ax_scatter.set_xlabel("W normalized")
	ax_scatter.set_ylabel("H normalized")
	ax_scatter.set_title("H-W plane with decision boudary for k-NN with k={}".format(k))
	fig_scatter.show()


	Metrics = calculate_performance(ybar, data["y_test"])
	fig, ax = plt.subplots()
	index = np.arange(len(Metrics))
	ax.bar(index, Metrics)
	ax.set_ylabel("Value")
	ax.set_title("Performance Metrics for KNN Classifier with K={}".format(k))
	ax.set_xticklabels(('', 'Hit Rate', 'Specificity','Sensitivity', 'PPV', 'NPV'))
	fig.show()


	input("pause")
if __name__ == "__main__":
	main()
