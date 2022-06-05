import copy
import random
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt



def lagrange(x, y):
    fun = np.zeros(x.shape[0])
    for point in range(x.shape[0]):
        xs = np.append(x[:point], x[point+1:])
        numerator = np.poly(xs)
        denominator = np.prod(x[point] - xs)
        fun += y[point] * (numerator / denominator)

    xp = np.linspace(x[0], x[-1], num=1000)
    plt.plot(xp, np.polyval(fun, xp))
    plt.scatter(x, y)
    plt.ylim([-50, max(y) + 50])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Interpolacja Lagrange dla {x.shape[0] - 1} przedziałów")
    plt.show()

def splines(x, y):
    k = x.shape[0] - 1
    A = np.zeros((4*k, 4*k))
    r = np.zeros((4*k, 1))
    for i in range(k):
        xn = x[i]
        xnn = x[i+1]
        A[(4*i), (4*i):(4*(i+1))] = np.array([xn**3, xn**2, xn, 1])
        r[(4*i)] = y[i]
        A[(4*i)+1, (4*i):(4*(i+1))] = np.array([xnn**3, xnn**2, xnn, 1])
        r[(4 * i)+1] = y[i + 1]
        if i != k - 1:
            A[(4*i) + 2, (4*i):(4*(i+2))] = np.array([3 * xnn**2, 2*xnn, 1, 0, -3 * xnn**2, -2*xnn, -1, 0])
            A[(4*i) + 3, (4*i):(4*(i+2))] = np.array([6 * xnn, 2, 0, 0, -6 * xnn, -2, 0, 0])
        else:
            A[(4*i) + 2, :2] = np.array([6 * x[0], 2])
            A[(4*i) + 3, (4*i):(4*i + 2)] = np.array([6 * x[-1], 2])
        r[(4*i)+2] = 0
        r[(4*i)+3] = 0
    coeffs = solve(A, r)

    for i in range(k):
        xp = np.linspace(x[i], x[i+1], num=10)
        coeff = coeffs[(4*i):4*(i+1)]
        plt.plot(xp, np.polyval(coeff, xp))
    plt.scatter(x, y)
    plt.ylim([-50, max(y) + 50])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Interpolacja Spline dla {k} przedziałów")
    plt.show()



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
    lagrange(x_sampled, y_sampled)
    splines(x_sampled, y_sampled)

def solve(A, B):
    n = A.shape[0]
    y = np.zeros((n, 1))
    ret = np.zeros((n, 1))

    U = copy.copy(A)
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        piv_idx = np.argmax(abs(U[k:, k])) + k
        L[[k, piv_idx], :] = L[[piv_idx, k], :]
        U[[k, piv_idx], :] = U[[piv_idx, k], :]
        P[[k, piv_idx], :] = P[[piv_idx, k], :]
        for j in range(k + 1, n):
            L[j, k] = U[j, k] / U[k, k]
            for i in range(k, n):
                U[j, i] = U[j, i] - (L[j, k] * U[k, i])

    b = P.dot(B)

    for i in range(n):
        s = 0
        for k in range(i):
            s += L[i, k] * y[k, 0]
        y[i] = b[i] - s

    for i in range(n-1, -1, -1):
        s = 0
        for k in range(i+1, n):
            s += U[i, k] * ret[k]
        ret[i] = (y[i] - s) / U[i, i]

    return ret


def main():
    df = read_csv("100.csv")
    x, y = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    interpolate(x, y, 15, "equal")

if __name__ == "__main__":
    main()
