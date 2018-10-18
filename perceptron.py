import numpy as np
import math
import matplotlib.pyplot as plt
from data_load import preprocessData
from knn import calculate_performance

def eval_Model(data, wi, wi0, mode):
	counter = 0
	y_hat = np.zeros(data["y_test"].shape)
	for k, x_k in enumerate(data["x_{}".format(mode)]):
		if np.dot(wi, x_k.T) > -wi0:
			y_hatk = 1
		else:
			y_hatk = 0
		y_hat.append(y_hatk)
		y_k = data["y_{}".format(mode)][k]
		if y_k == y_hatk:
			counter += 1  # Keep track of training accuracy
	print("{} Accuracy: {}".format(mode, counter/len(data["x_{}".format(mode)])))

	return calculate_performance(np.array(y_hat), data["y_{}".format(mode)])



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

	wi = np.array([[wi1, wi2]])

	eval_Model(data, wi, wi0, "train")
	fig, ax = plt.subplots()
	x1_plt = np.arange(0,1,1e-4)
	x2_plt = (-1./wi[0,1])*(x1_plt*wi[0,0]+wi0)
	ax.plot(x1_plt,x2_plt, color='b', linewidth=1.0)
	for epoch in range(epochs):
		# Training phase
		counter = 0
		if epoch == epochs-1:
			y_hat = []  # Store the y_hat values for the last training pass.
		for k, x_k in enumerate(data["x_train"]):
			if np.dot(wi, x_k.T) > -wi0:
				y_hatk = 1
			else:
				y_hatk = 0

			if epoch == epochs-1:
				y_hat.append(y_hatk)

			y_k = data["y_train"][k]
			if y_k == y_hatk:
				counter += 1  # Keep track of training accuracy

			wi = wi + LR * (y_k- y_hatk) * x_k
			wi0 = wi0 + LR * (y_k - y_hatk)
		if epoch % 20 == 0:
			print("Epoch: {} Training Accuracy: {:2f}".format(epoch, counter/k))

	# Evaluate the performance metrics after training on the training set
	Metrics_train = eval_Model(data, wi, wi0, "train")
	# " " on the testing set
	Metrics_test = eval_Model(data, wi, wi0, "test")

	# Plot performance metric bar charts
	"""
	fig, ax = plt.subplot	print(wi[0,0])
	print(wi[0,1])s(1,2)
	index = np.arange(len(Metrics_train))
	ax[0].bar(index, Metrics_train)
	ax[0].set_ylabel("Value")
	ax[0].set_title("Training Performance".format(LR))
	ax[0].set_xticklabels(('', 'Hit Rate', 'Specificity','Sensitivity', 'PPV', 'NPV'))
	ax[1].bar(index, Metrics_test)
	ax[1].set_ylabel("Value")
	ax[1].set_title("Test Performance".format(LR))
	ax[1].set_xticklabels(('', 'Hit Rate', 'Specificity','Sensitivity', 'PPV', 'NPV'))
	fig.suptitle("Performance Metrics for Perceptron Classifier with LR={}")
	plt.show()
	"""

	ax.scatter(data["x_train"][:,0], data["x_train"][:,1], c=data["y_train"])
	x1_plt = np.arange(0,1,1e-4)
	x2_plt = (-1./wi[0,1])*(x1_plt*wi[0,0]+wi0)
	ax.plot(x1_plt,x2_plt, color='r', linewidth=2.0)
	ax.set_xlim([0,1])
	ax.set_ylim([0,1])
	plt.show()

	return wi, wi0  # Return trained weights for plotting



def main():
	data = preprocessData()
	wi, wi0 = perceptron_train(data, 1000, 1e-3)

	# Move this outside of here. Return the weight array and plot outside b/c I need all data.
	fig, ax = plt.subplots()
	ax.scatter(data["x_train"][:,0], data["x_train"][:,1], c=data["y_train"])
	x1_plt = np.arange(0,1,1e-4)
	x2_plt = (-1./wi[0,1])*(x1_plt*wi[0,0]+wi0)
	ax.plot(x1_plt,x2_plt, color='r', linewidth=2.0)
	ax.set_xlim([0,1])
	ax.set_ylim([0,1])
	plt.show()


if __name__ == "__main__":
	main()
