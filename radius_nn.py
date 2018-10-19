import numpy as np
import math
import random
import matplotlib.pyplot as plt
from data_load import preprocessData, create_train_test_split
from knn import calculate_performance, knn

def rnn(data, R):

    # for each test data point,
    ybar = np.zeros(data["y_test"].shape)
    for i, x_test in enumerate(data["x_test"]):
        #loop through each training point.
        dist_list = []
        for j, x_train in enumerate(data["x_train"]):
            # compute euclidean distance between points
            dist = math.sqrt((x_test[0]-x_train[0])**2 + (x_test[1] - x_train[1])**2)

            if dist < R:
                dist_list.append([dist, data["y_train"][j]])

        one_cnt , zero_cnt= 0, 0
        for element in dist_list:
            if element[1] == 0:
                zero_cnt += 1
            else:
                one_cnt += 1

        if one_cnt == 0 and zero_cnt == 0:
            # No neighbors in radius, randomly assign.
            if random.randint(0,1) == 1:
                ybar[i] = 1
        elif one_cnt > zero_cnt:
            ybar[i] = 1  # Assume 0 otherwise

    return calculate_performance(ybar, data["y_test"])

def Sweep_Param(x, y, P_arr, R_K):

    means = []
    stds = []
    for P in P_arr:
        metrics_lst = []
        for i in range(10):
            data = create_train_test_split(x, y, 0.8)
            if R_K == 1:
                metrics_lst.append(rnn(data, P))
            else:
                metrics_lst.append(knn(data, P))


        means.append(np.mean(np.asarray(metrics_lst), axis=0))
        stds.append(np.std(np.asarray(metrics_lst), axis=0))


    # Each column is a graph of means vs. R for each performance metric.
    means = np.asarray(means)
    stds = np.asarray(stds)
    label_lst = ['Hit Rate', 'Specificity','Sensitivity', 'PPV', 'NPV']
    if R_K == 1:
        xlab = "Radius-R"
        title = "R"
    else:
        xlab= "K neighbors"
        title = "K"

    for i in range(5):
        fig, ax = plt.subplots()
        ax.errorbar(P_arr, means[:,i], yerr=stds[:,i], fmt='.k', capsize=5)
        ax.set_xlabel(xlab)
        ax.set_ylabel(label_lst[i])
        ax.set_title("{} vs. {}".format(label_lst[i], title))
        fig.show()

    input("pause")



def main():

    x, y = preprocessData()
    R_min = 0.01
    R_max = 0.3
    iterations = 10
    R_arr = np.linspace(R_min, R_max, iterations)

    Sweep_Param(x, y, R_arr, 1)



    # Pick best and redo to get the boundary plot

if __name__ == "__main__":
	main()
