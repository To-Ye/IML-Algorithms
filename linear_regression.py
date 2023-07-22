import numpy as np
import random
from matplotlib import pyplot as plt
from datageneration import *


def feature_generator(x):

    feature_vectors = []

    for x_i in x:
        feature_vectors.append(np.array([x_i**i for i in range(50)]))

    features = np.row_stack(feature_vectors)

    return features


def least_squares():
    lin_x, lin_y = gen_lin()

    n = np.shape(lin_x)[0]
    A = np.column_stack((lin_x, np.ones(n)))

    w = np.linalg.solve((A.conj().T @ A), A.conj().T @ lin_y)
    m_pred = w[0]
    b_pred = w[1]

    x = np.linspace(0, 1, 10)
    y_pred = x * m_pred + b_pred

    plt.scatter(lin_x, lin_y, label="LeastSquares(Data)")
    plt.plot(x, y_pred, label="LeastSquares")


def gradient_descent():
    lin_x, lin_y = gen_lin()

    n = np.shape(lin_x)[0]

    m_pred = random.random()
    b_pred = random.random()
    learning_rate = 0.01

    L = 1/n * np.sum((lin_y - (m_pred*lin_x + b_pred))**2)

    for i in range(1000):

        grad_L_m = (2/n) * np.sum((-lin_y + m_pred * lin_x + b_pred)*lin_x)
        grad_L_b = (2/n) * np.sum(-lin_y + m_pred * lin_x + b_pred)

        m_pred = m_pred - learning_rate * grad_L_m
        b_pred = b_pred - learning_rate * grad_L_b

    x = np.linspace(0, 1, 10)
    y_pred = x * m_pred + b_pred

    plt.scatter(lin_x, lin_y, label="GradientDescent(data)")
    plt.plot(x, y_pred, label="GradientDescent")


def least_squares_features():
    non_lin_x, non_lin_y = gen_non_lin()

    A = feature_generator(non_lin_x)

    n = np.shape(A)[0]

    w = np.linalg.solve((A.conj().T @ A), A.conj().T @ non_lin_y)

    x = np.linspace(0, 1, 100)

    y_pred = feature_generator(x) @ w

    plt.scatter(non_lin_x, non_lin_y, label="LeastSquaresFeatures(data)")
    plt.plot(x, y_pred, label="LeastSquaresFeatures")


def lasso_optimization():
    non_lin_x, non_lin_y = gen_non_lin()

    features = feature_generator(non_lin_x)

    n = np.shape(features)[0]
    m = np.shape(features)[1]
    w_pred = np.random.random(m) - 0.5

    learning_rate = 0.01
    regularization_term = 0.01

    L = (1 / n * np.sum((non_lin_y - features @ w_pred ) ** 2)) + (regularization_term * np.sum(np.abs(w_pred)))

    for i in range(1000):

        grad_L_w = regularization_term
        for i in range(n):
            grad_L_w += 2 / n * (features[i, :] @ w_pred - non_lin_y[i]) * features[i, :]

        w_pred = w_pred - learning_rate * grad_L_w

    x = np.linspace(0, 1, 100)
    y_pred = feature_generator(x) @ w_pred

    plt.scatter(non_lin_x, non_lin_y, label="Lasso(data)")
    plt.plot(x, y_pred, label="Lasso")


def ridge_closed_form():
    non_lin_x, non_lin_y = gen_non_lin()

    A = feature_generator(non_lin_x)

    n = np.shape(A)[0]
    m = np.shape(A)[1]

    regularization_term = 0.001
    I = np.identity(m)

    w = np.linalg.solve((A.conj().T @ A + regularization_term * I), A.conj().T @ non_lin_y)

    x = np.linspace(0, 1, 100)

    y_pred = feature_generator(x) @ w

    plt.scatter(non_lin_x, non_lin_y, label="RidgeClosedForm(data)")
    plt.plot(x, y_pred, label="RidgeClosedForm")


def ridge_optimization():
    non_lin_x, non_lin_y = gen_non_lin()

    A = feature_generator(non_lin_x)

    n = np.shape(A)[0]
    m = np.shape(A)[1]
    w_pred = np.random.random(m) - 0.5

    learning_rate = 0.1
    regularization_term = 0.01

    L = (1 / n * np.sum((non_lin_y - A @ w_pred) ** 2)) + (regularization_term * np.sum(np.abs(w_pred)))

    for _ in range(1000):

        grad_L_w = 2 * regularization_term * w_pred
        for i in range(n):
            grad_L_w += 1 / n * (A[i, :] @ w_pred - non_lin_y[i]) * A[i, :]

        w_pred = w_pred - learning_rate * grad_L_w

    x = np.linspace(0, 1, 100)
    y_pred = feature_generator(x) @ w_pred

    plt.scatter(non_lin_x, non_lin_y, label="RidgeOptimization(data)")
    plt.plot(x, y_pred, label="RidgeOptimization")


def main():
    # least_squares()
    # gradient_descent()
    least_squares_features()
    # lasso_optimization()
    # ridge_closed_form()
    ridge_optimization()

    plt.ylim(0,2)
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()