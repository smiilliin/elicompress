import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 1 / 27 * np.power(x, 3) - 3 / 9 * np.power(x, 2)


def elimf(a, z, x):
    y = np.zeros_like(x, dtype=float)
    l = len(z)
    for i in range(l):
        former = 1
        latter = np.ones_like(x, dtype=float)
        for j in range(1, l):
            former *= x - z[(i + j) % l].real
            latter *= z[i].real - z[(i + j) % l].real
        latter = a / l * (x - z[i].real) + z[i].imag / latter
        y += former * latter

    return y


def grad_a(z1, a, z2):
    grad = 0
    for z in z1:
        grad_t = 2 * (elimf(a, z2, z.real) - z.imag) * np.prod(z.real - z2.real)
        grad += grad_t

    return grad


x = np.arange(0, 10.1, 0.1)
S = x
S2 = [1, 3, 4]
z1 = S + f(S) * 1j
z2 = S2 + f(S2) * 1j
a = 2

plt.scatter(x, f(x))

for i in range(100):
    y = elimf(a, z2, x)
    print(grad_a(z1, a, z2))
    a -= 0.00001 * (1 / len(S)) * grad_a(z1, a, z2)

plt.plot(x, y)

plt.show()
