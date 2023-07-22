import numpy as np
import random
from matplotlib import pyplot as plt


def gen_lin(n=20, noise_level=0.1):

    m = random.random()
    b = random.random()

    x = np.random.random(n)
    y = m * x + b + (np.random.random(n) * noise_level)

    return x, y


def gen_non_lin(n=50, noise_level=0.1):

    root_1 = random.random()
    root_2 = random.random()
    root_3 = random.random()
    root_4 = random.random()
    root_5 = 1.1
    root_6 = - 0.1

    x = np.random.random(n)

    # y = ((x - root_1) * (x - root_2) + ((np.random.random(n)-0.5) * noise_level)) * 100
    y = (((x + 1) * (x - 1) * 0.25 + 1) + ((np.random.random(n) - 0.5) * noise_level))
    # y = ((x - root_1) * (x - root_2) * (x - root_3) * (x - root_4) * (x - root_5) * (x - root_6) + ((np.random.random(n) - 0.5) * noise_level)) * 100

    return x, y


def gen_lin_separable(n=20, noise_level=0.1):

    lin_x, lin_y = gen_lin(n)
    labels = np.zeros(n)
    moved_y = np.zeros(n)

    # for i in range(n):
    #     distance = np.random.random()-0.5
    #
    #     if distance == 0:
    #         distance = -1
    #
    #     labels[i] = np.sign(distance)
    #     moved_y[i] = lin_y[i]+distance
    #
    # count = 0
    # for y in labels:
    #     if y < 0:
    #         count += 1
    #
    # neg_y = np.zeros(count)
    # neg_x = np.zeros(count)
    # pos_y = np.zeros(n-count)
    # pos_x = np.zeros(n-count)

    # j = 0
    # k = 0
    # for i in range(n):
    #     if labels[i] < 0:
    #         neg_y[j] = moved_y[i]
    #         neg_x[j] = lin_x[i]
    #         j += 1
    #     else:
    #         pos_y[k] = moved_y[i]
    #         pos_x[k] = lin_x[i]
    #         k += 1
    #
    # print(neg_y)
    # print(pos_y)
    # return neg_y, neg_x, pos_y, pos_x

    return lin_x, moved_y, labels

def main():

    # lin_x, lin_y = gen_lin()
    # non_lin_x, non_lin_y = gen_non_lin()

    lin_x, lin_y, labels = gen_lin_separable(40)

    plt.scatter(lin_x, lin_y, label="LinearlySeparable")
    # plt.scatter(non_lin_x, non_lin_y, label="non_lin")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

