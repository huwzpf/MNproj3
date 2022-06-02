import random
from math import floor

from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt


def plot_poly(coeff, x, y):
    xp = np.linspace(x[0], x[-1], num=1000)
    plt.plot(xp, np.polyval(coeff, xp))
    plt.scatter(x, y)
    plt.ylim([-500, max(y)+500])
    plt.show()


def lagrange(x, y):
    fun = np.zeros(x.shape[0])
    for point in range(x.shape[0]):
        xs = np.append(x[:point], x[point+1:])
        numerator = np.poly(xs)
        denominator = np.prod(x[point] - xs)
        fun += y[point] * (numerator / denominator)
    return fun


def interpolate(x, y, k, interval="equal"):
    if interval == "random":
        idx = random.sample(range(1, x.shape[0]-1), k-1)
        idx += [0, x.shape[0]-1]
        idx.sort()
        x_sampled = np.array([x[i] for i in idx])
        y_sampled = np.array([y[i] for i in idx])
    else:
        if interval != "equal":
            print("choosing equal interval size")
        w = x.shape[0] // k
        if (x.shape[0]-1) % k != 0:
            print(f"cant equally divide x in {k} intervals")

        x_sampled = x[::w]
        y_sampled = y[::w]
    plot_poly(lagrange(x_sampled, y_sampled), x_sampled, y_sampled)




def main():
    df = read_csv("100.csv")
    x, y = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    interpolate(x, y, 15, "equal")
if __name__ == "__main__":
    main()
