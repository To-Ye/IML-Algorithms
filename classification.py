import numpy as np
import random
from matplotlib import pyplot as plt
from datageneration import *


def linear_classifier():
    X, labels, neg_x, neg_y, pos_x, pos_y = gen_lin_separable()

    n = np.shape(X)[0]
    m = np.shape(X)[1]
    w_pred = np.random.random(n) - 0.5
    b = random.random()

    learning_rate = 0.01

    L = (w_pred.T @ X + b) @ (-labels)

    for _ in range(1000):

        predictions = w_pred.T @ X + b
        errors = (labels-predictions)

        grad_L_w = -X @ errors
        grad_L_b = -np.sum(errors)

        w_pred = w_pred - learning_rate * grad_L_w
        b = b - learning_rate * grad_L_b

        print(np.linalg.norm(w_pred)**2 + b**2)

    x = np.linspace(0, 1, 100)
    m = -w_pred[0] / w_pred[1]
    y_pred = x * m + b

    plt.scatter(neg_x, neg_y, label="neg(data)")
    plt.scatter(pos_x, pos_y, label="pos(data)")
    plt.plot(x, y_pred, label="LinearClassification")


def main():
    linear_classifier()

    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()