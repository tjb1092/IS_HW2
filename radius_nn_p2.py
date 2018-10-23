import numpy as np
import math
import random
import matplotlib.pyplot as plt
from data_load import preprocessData, create_train_test_split
from knn import calculate_performance, knn

def radius_nn(data, R):

    # for each test data point,
    ybar = np.zeros((1,len(data["x_test"])))
    for i, x_test in enumerate(data["x_test"]):
        #loop through each training point.
        dist_list = []
        for j, x_train in enumerate(data["x_train"]):
            # compute euclidean distance between points
            dist = math.sqrt((x_test[0]-x_train[0])**2 + (x_test[1] - x_train[1])**2)

            if dist < R:
                # If point is within radius R, add to neighbor list
                dist_list.append([dist, data["y_train"][j]])

        one_cnt , zero_cnt= 0, 0
        # Add the total 1's, and 0's from the point's neighbors
        for element in dist_list:
            if element[1] == 0:
                zero_cnt += (1./element[0])
            else:
                one_cnt += (1./element[0])

        if one_cnt == 0 and zero_cnt == 0:
            # No neighbors in radius, randomly assign.
            if random.randint(0,1) == 1:
                ybar[0,i] = 1
        # Has a neighbor, compare computed score for the classes
        elif one_cnt > zero_cnt:
            ybar[0,i] = 1  # Assume 0 otherwise
    # Return performance metric
    return ybar

def Sweep_Param(x, y, P_arr, P_best, R_K):
    means = []
    stds = []
    for P in P_arr:
        # For each value of the parameter,
        metrics_lst = []
        for i in range(10):
            # Run 10 independent trials w/ new data each time
            data = create_train_test_split(x, y, 0.8)
            # Select either radius-NN or k-NN.
            if R_K == 1:
                ybar = radius_nn(data, P)
            else:
                ybar = knn(data,P)

            metrics_lst.append(calculate_performance(ybar, data["y_test"]))

        # Compute the mean and standard deviation for each metric for each set of 10 trials
        means.append(np.mean(np.asarray(metrics_lst), axis=0))
        stds.append(np.std(np.asarray(metrics_lst), axis=0))


    # Each column is a graph of means vs. R for each performance metric.
    means = np.asarray(means)
    stds = np.asarray(stds)
    label_lst = ['Hit Rate', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
    if R_K == 1:
        xlab = "Radius-R"
        title = "R"
    else:
        xlab= "k neighbors"
        title = "k"

    for i in range(5):
        # For each metric, index the proper column of data and plot
        fig, ax = plt.subplots(figsize=(6,6))
        ax.errorbar(P_arr, means[:,i], yerr=stds[:,i], fmt='.k', capsize=5)
        ax.set_xlabel(xlab)
        ax.set_ylabel(label_lst[i])
        ax.set_title("{} vs. {}".format(label_lst[i], title))
        fig.show()


    # Pick best one and redo to plot decision boundary
    data = create_train_test_split(x, y, 0.8)  # Create splits

    fig_scatter, ax_scatter = plt.subplots(figsize=(6,6))  # Get data for H-W Plane

    if R_K == 1:
        ybar = radius_nn(data, P_best)
    else:
        ybar = knn(data, P_best)

    # Plot the W-H data colored by class.
    labels = ["Unstressed", "Stressed"]
    colors = ["b", "r"]
    for i in range(2):
        plt_index = data["y_train"] == i
        plt_test_index = data["y_test"] == i
        ax_scatter.scatter(data["x_train"][plt_index,0], data["x_train"][plt_index,1], c=colors[i], s=10, marker="D", label="{} - Train".format(labels[i]))
        ax_scatter.scatter(data["x_test"][plt_test_index,0], data["x_test"][plt_test_index,1], c=colors[i], s=60, marker="x", label="{} - Test".format(labels[i]))

    # Create a test vector to plot the decision boundary.
    d_testx = np.arange(0,1.01,1e-2)
    d_test_grid = np.array(np.meshgrid(d_testx, d_testx)).T.reshape(-1,2)  # Create grid of test-points
    data["x_test"] = d_test_grid  # Change over X's to test grid, don't care about y's

    if R_K == 1:
        ybar_grid = radius_nn(data, P_best)
        title=("Radius","R")
    else:
        ybar_grid = knn(data, P_best)
        title=("k","k")


    # Plot the deicison boundary
    labels = ["Unstressed", "Stressed"]
    colors = ["b", "r"]
    for i in range(2):
        plt_index = ybar_grid[0,:] == i
        ax_scatter.scatter(data["x_test"][plt_index,0], data["x_test"][plt_index,1], c=colors[i], s=0.25)

    ax_scatter.set_xlabel("W normalized")
    ax_scatter.set_ylabel("H normalized")
    ax_scatter.set_title("Best {}-NN decision boundary with {}={}".format(title[0], title[1], P_best))
    ax_scatter.legend()
    plt.show()


def main():

    x, y = preprocessData()  # Load data
    # Radius range
    R_min = 0.01
    R_max = 0.3
    iterations = 10  #Number of parameter to try in range.
    R_best = 0.14
    # Create parameter array of Rs to test
    R_arr = np.linspace(R_min, R_max, iterations)
    print(R_arr)

    Sweep_Param(x, y, R_arr, R_best, 1)  # Get the performance for each value of R to try.

if __name__ == "__main__":
	main()
