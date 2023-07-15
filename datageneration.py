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


def main():

    lin_x, lin_y = gen_lin()
    non_lin_x, non_lin_y = gen_non_lin()

    plt.scatter(lin_x, lin_y, label="lin")
    plt.scatter(non_lin_x, non_lin_y, label="non_lin")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

