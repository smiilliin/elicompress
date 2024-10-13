import numpy as np
import matplotlib.pyplot as plt


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
    grad /= len(z1)

    if np.abs(grad) > 5:
        return 5 * np.abs(grad) / grad

    return grad


def errf(z1, a, z2):
    err = 0
    for z in z1:
        err += (elimf(a, z2, z.real) - z.imag) ** 2

    return err / len(z1)


def update_z2(z1, a, z2):
    updated_z2 = z2.copy()

    for i, z in enumerate(z2):
        left = np.where(z1.real < z.real, z.real - z1.real, np.inf)
        right = np.where(z1.real > z.real, z1.real - z.real, np.inf)

        if left.size == 0 or right.size == 0:
            continue

        li = left.argmin()
        ri = right.argmin()

        lz2 = z2.copy()
        rz2 = z2.copy()

        if not np.isin(updated_z2, z1[li]).any() and not np.isin(z1, z1[li]).any():
            lz2[i] = z1[li]
        if not np.isin(updated_z2, z1[ri]).any() and not np.isin(z1, z1[ri]).any():
            rz2[i] = z1[ri]

        erri = np.array([errf(z1, a, z2), errf(z1, a, lz2), errf(z1, a, rz2)]).argmin()

        if erri == 0:
            updated_z2[i] = z
        elif erri == 1:
            updated_z2[i] = lz2[i]
        elif erri == 2:
            updated_z2[i] = rz2[i]

    return updated_z2


x = np.arange(0, 20.1, 0.1)
S = np.arange(0, 20.1, 1)
z1 = S + np.sin(np.random.normal(0, np.pi / 4, len(S))) * 1j

I = np.arange(0, 21, 3, dtype=int)
S2 = S[I]
z2 = S2 + z1[I].imag * 1j

a = 0
r = 0.00000001

plt.ylim((-1, 1))

epochs = 150

for i in range(epochs):
    y = elimf(a, z2, x)
    print(i, a, errf(z1, a, z2))

    new_z2 = update_z2(z1, a, z2)
    new_a = a - r * grad_a(z1, a, z2)

    z2 = new_z2
    a = new_a

plt.plot(x, y, color="blue")

plt.scatter(z1.real, z1.imag)
plt.scatter(z2.real, z2.imag)

plt.plot(x, y)

plt.show()
