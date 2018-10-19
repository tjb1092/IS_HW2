import numpy as np
import math
import matplotlib.pyplot as plt
from data_load import preprocessData, create_train_test_split
from knn import calculate_performance

def eval_Model(data, wi, wi0, mode, metrics):
	counter = 0
	y_hat = np.zeros(data["y_{}".format(mode)].shape)
	for k, x_k in enumerate(data["x_{}".format(mode)]):

		if np.dot(wi, x_k.T) > -wi0:
			y_hatk = 1
		else:
			y_hatk = 0
		y_hat[k] = y_hatk

		y_k = data["y_{}".format(mode)][k]
		if y_k == y_hatk:
			counter += 1  # Keep track of training accuracy

	print("{} Hit Rate: {}".format(mode, counter/len(data["x_{}".format(mode)])))

	if metrics:
		return calculate_performance(np.array(y_hat), data["y_{}".format(mode)])
	else:
		# 1 - (Correct guesses / total number of samples).
		return 1 - (counter/len(data["x_{}".format(mode)]))



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
	# perpendicular to a point from both classes
	wi0 = -1.0
	wi2 = (2.0 * (p2[0] - p1[0])) / ((p1[1] - p2[1]) * (p1[0] + p2[0]) - (p1[1] + p2[1]) * (p2[0] - p1[0]))
	wi1 = ((p2[1] - p1[1]) / (p2[0] - p1[0])) * wi2

	wi = np.array([[-wi1, -wi2]])

	x1_plt_init = np.arange(0,1,1e-4)
	x2_plt_init = (-1./wi[0,1])*(x1_plt_init*wi[0,0]+wi0)

	epoch_arr = []
	err_rate_train = []
	err_rate_test = []

	# Initialize best weight trackers
	wi_best = wi
	wi0_best = wi0
	best_err = 1
	best_epoch = 0

	for epoch in range(epochs):
		# Training phase
		counter = 0
		acc_wi = np.zeros(wi.shape)
		acc_w0 = 0
		for k, x_k in enumerate(data["x_train"]):
			if  np.dot(wi, x_k.T) > -wi0:
				y_hatk = 1
			else:
				y_hatk = 0

			y_k = data["y_train"][k]
			if y_k == y_hatk:
				counter += 1  # Keep track of training accuracy

			wi = wi + LR * (y_k - y_hatk) * x_k
			wi0 = wi0 + LR * (y_k - y_hatk)
			x2_plt_2 = (-1./wi[0,1])*(x1_plt_init*wi[0,0]+wi0)

		#wi = wi + acc_wi
		#wi0 = wi0 + acc_w0
		if epoch % 50 == 0:
			print(wi, wi0)

			epoch_arr.append(epoch)
			err_rate_train.append(eval_Model(data, wi, wi0, "train", 0))
			err_rate_test.append(eval_Model(data, wi, wi0, "test", 0))

		if err_rate_test[-1] < best_err:
			wi_best = wi
			wi0_best = wi0
			best_err = err_rate_test[-1]
			best_epoch = epoch

	fig, ax = plt.subplots()

	print("Best Error: {} \n Best Epoch: {}".format(best_err, best_epoch))
	ax.scatter(data["x_train"][:,0], data["x_train"][:,1], c=data["y_train"], label="Training Data")
	x1_plt_trained = np.arange(0,1,1e-4)
	x2_plt_trained = (-1./wi_best[0,1])*(x1_plt_trained*wi_best[0,0]+wi0_best)

	ax.plot(x1_plt_init,x2_plt_init, color='b', label="Initial Boundary", linewidth=1.0)

	ax.plot(x1_plt_trained,x2_plt_trained, color='r', label="Trained Boundary", linewidth=2.0)
	ax.set_xlim([0,1])
	ax.set_ylim([0,1])
	ax.legend()
	fig.show()

	fig, ax = plt.subplots()
	ax.plot(epoch_arr, err_rate_train, label="Training Error Rate")
	ax.plot(epoch_arr,err_rate_test, label="Test Error Rate")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Error Rate")
	ax.set_title("Error Rates per Epoch")
	ax.legend()
	fig.show()

	# Evaluate the performance metrics after training on the training set
	Metrics_train = eval_Model(data, wi_best, wi0_best, "train", 1)
	# " " on the testing set
	Metrics_test = eval_Model(data, wi_best, wi0_best, "test", 1)

	# Plot performance metric bar charts
	fig, ax = plt.subplots(1,2)
	index = np.arange(len(Metrics_train))
	ax[0].bar(index, Metrics_train)
	ax[0].set_ylabel("Value")
	ax[0].set_ylim([0, 1])
	ax[0].set_title("Training Performance".format(LR))
	ax[0].set_xticklabels(('', 'Hit Rate', 'Specificity','Sensitivity', 'PPV', 'NPV'))
	ax[1].bar(index, Metrics_test)
	ax[1].set_ylabel("Value")
	ax[1].set_ylim([0, 1])
	ax[1].set_title("Test Performance".format(LR))
	ax[1].set_xticklabels(('', 'Hit Rate', 'Specificity','Sensitivity', 'PPV', 'NPV'))
	fig.suptitle("Performance Metrics for Perceptron Classifier with LR={}")
	plt.show()

	return wi_best, wi0_best  # Return trained weights for plotting



def main():
	LR = 1e-2
	x, y = preprocessData()
	data = create_train_test_split(x, y, 0.8)

	wi, wi0 = perceptron_train(data, 2000, LR)

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
