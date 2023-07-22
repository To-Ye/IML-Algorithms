import numpy as np
import random
from matplotlib import pyplot as plt
from datageneration import *

def k_means():

    k = 2
    n = 50

    K_n = gen_clusters(n,k)

    centers = np.random.random((2,k))

    for _ in range(10):

        clusters = []

        for _ in range(k):
            clusters.append([])

        for i in range(k*n):
            distance = []
            current_point = K_n[:,i]

            for j in range(k):
                distance.append(np.linalg.norm(current_point - centers[:,j]))

            clusters[np.argmin(distance)].append(current_point)

        for j in range(k):
            centers[:,j] = np.mean(clusters[j], axis=0)

    plt.scatter(K_n[0,:], K_n[1,:])
    plt.scatter(centers[0,:], centers[1,:], label="centers", marker="x", c="red")



def main():
    k_means()

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()