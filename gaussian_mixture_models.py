import numpy as np
import random
from matplotlib import pyplot as plt


def soft_e_step(mu_one, mu_zero, pos, D):

    pies = []
    number_of_participants = len(D)

    for i in range(number_of_participants):
        pie = ((mu_one**D[i][0]) * (mu_zero**D[i][1]) * pos) / \
              ((mu_zero**D[i][0]) * (mu_one**D[i][1]) * (1-pos) + ((mu_one**D[i][0]) * (mu_zero**D[i][1]) * pos))
        pies.append(pie)

    return pies


def soft_m_step(pies, D):

    number_of_tests = D[0][0] + D[0][1]

    mu_one = (sum(pie*d[0] for pie, d, in zip(pies,D))) / (np.sum(pies) * number_of_tests)
    mu_zero = (sum((1 - pie) * d[0] for pie, d, in zip(pies, D))) / (sum(1 - pie for pie in pies) * number_of_tests)

    return mu_one, mu_zero


def soft_expectaction_maximisation():
    D = [(3, 2), (1, 4), (3, 2), (5, 0)]
    mu_one = 0.8
    mu_zero = 0.2
    pos = 0.2
    pies = []

    iterations = 1

    for i in range(iterations):
        pies = soft_e_step(mu_one, mu_zero, pos, D)
        mu_one, mu_zero = soft_m_step(pies, D)

    print("mu_one_{}: {}".format(iterations, mu_one))
    print("mu_zero_{}: {}".format(iterations, mu_zero))
    print("pos: {}".format(pos))
    print("pies: {}".format(pies))


def main():
    soft_expectaction_maximisation()


if __name__ == '__main__':
    main()