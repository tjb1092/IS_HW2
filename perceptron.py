import numpy as np
import math
import matplotlib.pyplot as plt
from data_load import preprocessData, create_train_test_split
from knn import calculate_performance

def eval_Model(data, wi, wi0, mode, metrics):
	# Can evaluate entire data set in one dot product operation
	y_hat = 1*(np.dot(wi, data["x_{}".format(mode)].T) > -wi0)
	y_k = data["y_{}".format(mode)]
	correct = np.sum(1*(y_hat == y_k))

	Hit_Rate = correct/len(data["x_{}".format(mode)])
	print("{} Hit Rate: {}".format(mode, Hit_Rate))

	if metrics:
		return calculate_performance(y_hat, data["y_{}".format(mode)])
	else:
		# Error Rate = 1-Hit_Rate
		return 1 - Hit_Rate

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

	#Negates all of the wieghts so that the decision boundary comparison works properly.
	wi = np.array([[-wi1, -wi2]])

	# record initial boundary line
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
		# For each epoch, loop through each data point and update weights
		counter = 0
		for k, x_k in enumerate(data["x_train"]):
			# Evaluate predicted class label
			if  np.dot(wi, x_k.T) > -wi0:
				y_hatk = 1
			else:
				y_hatk = 0

			y_k = data["y_train"][k]
			if y_k == y_hatk:
				counter += 1  # Keep track of training accuracy

			# Weight update policy
			wi = wi + LR * (y_k - y_hatk) * x_k
			wi0 = wi0 + LR * (y_k - y_hatk)

		if epoch % 50 == 0:
			print(wi, wi0)

			epoch_arr.append(epoch)
			err_rate_train.append(eval_Model(data, wi, wi0, "train", 0))
			err_rate_test.append(eval_Model(data, wi, wi0, "test", 0))
		# Check to see if newest test error is better than previous epoch
		if err_rate_test[-1] < best_err:
			# If so, then save weights and error rate
			wi_best = wi
			wi0_best = wi0
			best_err = err_rate_test[-1]
			best_epoch = epoch

	print("Best Error: {} \n Best Epoch: {}".format(best_err, best_epoch))
	fig, ax = plt.subplots(figsize=(6,6))
	# Plot training and test data
	labels = ["Unstressed", "Stressed"]
	colors = ["b", "r"]
	for i in range(2):
		plt_index = data["y_train"] == i
		plt_test_index = data["y_test"] == i
		ax.scatter(data["x_train"][plt_index,0], data["x_train"][plt_index,1], c=colors[i], s=20, label="{} - Train".format(labels[i]))
		ax.scatter(data["x_test"][plt_test_index,0], data["x_test"][plt_test_index,1], c=colors[i], s=60, marker="x", label="{} - Test".format(labels[i]))


	# Calculate points on trained boundary line
	x1_plt_trained = np.arange(0,1,1e-4)
	x2_plt_trained = (-1./wi_best[0,1])*(x1_plt_trained*wi_best[0,0]+wi0_best)
	# Plot initial and trained boundary lines
	ax.plot(x1_plt_init,x2_plt_init, color='y', label="Initial Boundary", linewidth=1.0)
	ax.plot(x1_plt_trained,x2_plt_trained, color='g', label="Trained Boundary", linewidth=2.0)
	ax.set_xlim([0,1])
	ax.set_ylim([0,1])
	ax.legend()
	ax.set_xlabel("W normalized")
	ax.set_ylabel("H normalized")
	ax.set_title("H-W plane with decision boundary for Perceptron with $\eta$={}".format(LR))
	fig.show()
	# Plot error rate per epoch
	fig, ax = plt.subplots(figsize=(6,6))
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
	fig, ax = plt.subplots(1,2, figsize=(12,6))
	index = np.arange(len(Metrics_train))
	ax[0].bar(index, Metrics_train)
	ax[0].set_ylabel("Value")
	ax[0].set_ylim([0, 1])
	ax[0].set_title("Training Performance".format(LR))
	ax[0].set_xticklabels(('', 'Hit Rate', 'Sensitivity', 'Specificity', 'PPV', 'NPV'))
	ax[1].bar(index, Metrics_test)
	ax[1].set_ylabel("Value")
	ax[1].set_ylim([0, 1])
	ax[1].set_title("Test Performance".format(LR))
	ax[1].set_xticklabels(('', 'Hit Rate', 'Sensitivity', 'Specificity', 'PPV', 'NPV'))
	fig.suptitle("Performance Metrics for Perceptron Classifier with $\eta$={}".format(LR))
	plt.show()

	return wi_best, wi0_best  # Return trained weights for plotting



def main():
	LR = 1e-2
	Epochs = 2000
	x, y = preprocessData()  # Load Data
	data = create_train_test_split(x, y, 0.8)

	# Train weights using perceptron algorithm
	wi, wi0 = perceptron_train(data, Epochs, LR)

if __name__ == "__main__":
	main()
